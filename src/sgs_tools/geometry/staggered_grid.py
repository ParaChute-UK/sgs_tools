import xarray as xr
from numpy import nan
from xarray.core.types import T_Xarray


def get_grid_spacing_coord(coord: xr.DataArray, new_dim: str) -> xr.DataArray:
    """get a coordinate for the grid spacing

    :param coord: dataarray of a coordinate
    :param new_dim: name of dimesion associated with grid spacing
            (would typically shift _face <-> _center)
    :return: The coordinate spacing with dimension `new_dim`
    """
    current_name = coord.dims[0]
    return coord.diff(dim=current_name, n=1).rename({current_name: new_dim})


def interpolate_to_grid(
    ds: T_Xarray,
    target_dims: list[str] = [],
    coord_map: dict[str, xr.DataArray] = {},
    drop_coords: bool = True,
) -> T_Xarray:
    """Spatial interpolation to a target_grid

    :param ds: input data array or dataset. Needs to have dimensions with coordinates
        that are labelled 'x*', 'y*', 'z*' etc. or

    :param target_dims: list of dimension names  to interpolate to, in the order xdim, ydim, zdim.
        They must exist in ds as DataArry/coordinates
    :param coord_map: dictionary of {existing_dimension_in_ds : target_coordinate_as_DataArray}
    :param drop_coords: flag to exclude spatial coordinates relying on removed dims from output
    :return: the input data interpolated to the target coordinates
    """
    if isinstance(target_dims, str):
        target_dims = [target_dims]

    if target_dims:
        assert not coord_map, "Should specify only target_dims or coord_map"
        # assume spatial coordinates are strings that start with x|y|z
        x_dims = [x for x in ds.dims if str(x).startswith("x")]
        y_dims = [y for y in ds.dims if str(y).startswith("y")]
        z_dims = [z for z in ds.dims if str(z).startswith("z")]
        missing_coords = [dim for dim in target_dims if dim not in ds.coords]
        assert (
            len(missing_coords) == 0
        ), f"missing target coordinages {missing_coords} from input data"
        x_target = [x for x in target_dims if x.startswith("x")]
        y_target = [y for y in target_dims if y.startswith("y")]
        z_target = [z for z in target_dims if z.startswith("z")]
        # check that there is at most one of {x|y|z}_target dimension
        assert len(x_target) <= 1
        assert len(y_target) <= 1
        assert len(z_target) <= 1

        coord_map = {}
        if len(x_target) == 1:
            coord_map.update(
                {str(x): ds[x_target[0]] for x in x_dims if str(x) != x_target[0]}
            )
        if len(y_target) == 1:
            coord_map.update(
                {str(y): ds[y_target[0]] for y in y_dims if str(y) != y_target[0]}
            )
        if len(z_target) == 1:
            coord_map.update(
                {str(z): ds[z_target[0]] for z in z_dims if str(z) != z_target[0]}
            )

    else:
        assert coord_map, "Should specify one of target_dims or coord_map"
        missing_dims = [dim for dim in coord_map.keys() if dim not in ds.dims]
        assert (
            len(missing_dims) == 0
        ), f"missing input dimensions {missing_dims} from input data"

    # rename any possible dimensions that will clash with coord-map targets:
    ds_interp = ds
    c_map = {}
    for d, c in coord_map.items():
        if c.name == d:
            ds_interp = ds_interp.rename({c.name: f"{c.name}_original"})
            c_map[f"{c.name}_original"] = coord_map[d]
        else:
            c_map[d] = coord_map[d]

    # drop any coordinates based on replaced dimensions
    drop_coords_list = []
    if drop_coords:
        for coord in ds_interp.coords:
            dims = ds_interp[coord].dims
            if dims and dims[0] in c_map:
                drop_coords_list.append(dims[0])

    # interpolate all fields to target grid and drop coordinates
    ds_interp = ds_interp.interp(c_map, method="linear", assume_sorted=True).drop_vars(
        drop_coords_list
    )
    return ds_interp


def compose_vector_components_on_grid(
    components: list[xr.DataArray],
    target_dims: list[str] = [],
    vector_dim: str = "c1",
    name: str = "",
    long_name: str = "",
    drop_coords: bool = True,
) -> xr.DataArray:
    """turn a list of arrays into a vector field
    :param components: list of vector components
    :param target_dims: if target_dims is given, it will interpolate onto it first, otherwise all componets must have the same dimesions and coordinates
    :param vector_dim: label of dimension indexing vector components
    :param name: name of new array
    :param lon_name: long_name of new array
    :param drop_coords: flag picked up by :meth:`interpolate_to_grid` to exclude spatial coordinates relying on removed dims from output
    """
    # interpolate
    if target_dims is []:
        assert all([components[0].dims == x.dims for x in components[1:]]), (
            "The components' dimensions don't match. "
            "Choose a set of dimensions to interpolate to!"
        )
        vec_arr = xr.concat(
            components,
            dim=xr.DataArray(range(1, len(components) + 1), dims=[vector_dim]),
        )
    else:
        assert all([c.name for c in components])
        vec_ds = xr.Dataset({c.name: c for c in components})
        vec_arr = interpolate_to_grid(
            vec_ds, target_dims, drop_coords=drop_coords
        ).to_array(dim="vel")
        vec_arr = vec_arr.assign_coords(
            {vector_dim: ("vel", range(1, len(components) + 1))}
        ).swap_dims({"vel": vector_dim})
    # combine into a vector
    # add meta data
    if name:
        vec_arr.name = name
    if long_name:
        vec_arr.attrs["long_name"] = long_name
    return vec_arr


def diff_lin_on_grid(
    ds: xr.DataArray, dim: str, periodic_field: bool = False
) -> xr.DataArray:
    """differentiate a dataarray on staggered grid
    assumes that we have coordinate staggering as\:

    .. parsed-literal::

        c_face(i) --  +  ------ c_face(i+1) -------- +
        |             |                 |            |
        + ------ c_centre(i) ---------- + ----- c_centre(i+1)

    where `c_{face|center}` is any dimension
    :param ds: input array to be differentiated
    :param dim: dimension along which to differentiate
    :param periodic_field: boundary condition treatment; if `False` will
    fill boundary values with NaN. Note that only 1 boundary is undefined\:
    upper boundary for face-coordinates, and lower boundary for centre-coordinates.
    :return: the derivative on the grid with offset staggering in the differentiated dimension
    face -> centre and v.v.
    """

    def delta(coord, shift):
        """constant extrapolation"""
        if shift == 1:
            coord_n1 = 2 * coord[0] - coord[1]
            return coord - coord.shift({coord.dims[0]: shift}, fill_value=coord_n1)
        elif shift == -1:
            coord_Np1 = 2 * coord[-1] - coord[-2]
            return coord - coord.shift({coord.dims[0]: shift}, fill_value=coord_Np1)

    def centre_point(coord, shift):
        if shift == 1:
            extra = coord[0] - (coord[1] - coord[0])
        elif shift == -1:
            extra = coord[-1] + (coord[-1] - coord[-2])
        return (coord + coord.shift({coord.dims[0]: shift}, fill_value=extra)) / 2

    coord = ds[dim].astype(float)  # just in case
    val = ds.astype(float)
    if dim.endswith("_face"):
        new_dim = dim.rstrip("_face") + "_centre"
        shift = 1
    elif dim.endswith("_centre"):
        new_dim = dim.rstrip("_centre") + "_face"
        shift = -1
    else:
        raise ValueError(f"Unrecognizable coordinate staggering for dimension {dim}")

    new_coord = centre_point(coord, shift)

    if periodic_field:
        deriv = (val - val.roll({dim: shift})) / delta(coord, shift)
    else:
        deriv = (val - val.shift({dim: shift}, fill_value=nan)) / delta(coord, shift)
    deriv = deriv.rename({dim: new_dim})
    deriv[new_dim] = new_coord.rename({dim: new_dim})
    for c in deriv.coords:
        if deriv[c].dims and deriv[c].dims[0] == new_dim and not c == new_dim:
            deriv[c] = centre_point(deriv[c], shift)
    return deriv


def grad_on_cart_grid(
    ds: xr.DataArray,
    space_dims: list[str],
    periodic_field: list[bool] = [False, False, False],
) -> xr.Dataset:
    """differentiate a scalar with respect to given space dims on staggered grid
    :param ds: input array to be differentiated
    :param space_dims: labels for the spatial dimensions (to be differentiated against)
    :param periodic_field: boundary condition treatment; passed to :meth:`diff_lin_on_grid`
    """
    name = ds.name
    if name is None:
        name = "d"
    grad = []
    for i, d in enumerate(space_dims):
        # label should turn into "{name}_dx", etc.
        pd = diff_lin_on_grid(ds, d, periodic_field[i])
        pd.name = f"{name}_{d[0]}"
        grad.append(pd)
    # merge with compat=minimal to drop conflicting non-dimension coordinates
    # with the same name but on staggered dimensions
    # e.g. d_x.x_cu(x_face) and d_y.x_cu(x_centre) dims
    return xr.merge(grad, compat="minimal")


def grad_vec_on_grid(
    ds: xr.Dataset,
    target_dims: list[str] = ["x_centre", "y_centre", "z_centre"],
    new_dim_name: list[str] = ["c1", "c2"],
    name: str | None = None,
) -> xr.DataArray:
    """computes gradient of a vector described onto target dimensions
    ds: should be a dataset which only contains the components of the vector in sorted order and target coordinates.
    target_dims: the dimensions to compute the derivative on (must be coordinates in input dataset)
    new_dim_name: the names of the new dimensions: [vector component, differential component]
    name : name for output dataarray (optional)
    """
    # unpack new_dim_name
    vec_name, d_name = new_dim_name
    gradvec_comp = {}

    for i, f in enumerate(ds):
        # individual sapce dimensions for each staggered field
        space_dims = sorted([str(d) for d in ds[f].dims if str(d)[0] in "xyz"])
        grad_f = grad_on_cart_grid(ds[f], space_dims)
        # interpolate onto target coordinates from original ds (must exist)
        # careful grad_on_cart_grid will create coordinates with possibly classhing names
        # so make an explicit coordinate map
        coord_map = {}
        for k in grad_f.dims:
            k_str = str(
                k
            )  # xr.Dataset.dims is a Hashable and interpolate expects a str
            if k_str[0] in "xyz":
                # name match by first character xyz
                target_coord = [c for c in target_dims if c[0] == k_str[0]][0]
                coord_map[k_str] = ds[target_coord]
        grad_f_on_cent_ds = interpolate_to_grid(grad_f, coord_map=coord_map)
        return grad_f_on_cent_ds
        # convert to a dataarray
        grad_f_on_cent = grad_f_on_cent_ds.to_dataarray(d_name).sortby(d_name)
        # rename c1 coordinates: fragile this works only because of
        # the naming convention in interpolate_to_grid
        grad_f_on_cent[d_name] = [f"d{x.item()[-1]}" for x in grad_f_on_cent[d_name]]
        grad_f_on_cent[d_name] = [f"d{x.item()[-1]}" for x in grad_f_on_cent[d_name]]
        gradvec_comp[f] = grad_f_on_cent

    # gradvec_da = xr.Dataset(gradvec_comp).to_dataarray(dim=vec_name)
    # gradvec_da = gradvec_da.sortby(vec_name)
    # # rename c2 coordinates: less fragile but still a bit idiosyncratic
    # gradvec_da[vec_name] = [f"v{i+1}" for i in range(len(gradvec_comp))]
    # if name is not None:
    #     gradvec_da.name = name
    # return gradvec_da
