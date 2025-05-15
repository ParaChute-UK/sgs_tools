from dataclasses import dataclass


@dataclass(frozen=True)
class field_plot_kwargs:
    """a collection of field-specific plotting decorations"""

    label: str
    tcoord: str
    zcoord: str
    hcoords: tuple[str, str]
    vrange: tuple[float | None, float | None]
    cmap: str

    def with_args(self, **kwargs):
        pars = {
            "label": self.label,
            "tcoord": self.tcoord,
            "zcoord": self.zcoord,
            "hcoords": self.hcoords,
            "vrange": self.vrange,
            "cmap": self.cmap,
        }
        pars.update(kwargs)
        return field_plot_kwargs(**pars)


field_plot_map = {
    "u": field_plot_kwargs(
        "u", "t_0", "z_rho", ("x_face", "y_centre"), (None, None), "PiYG"
    ),
    "v": field_plot_kwargs(
        "v", "t_0", "z_rho", ("x_centre", "y_face"), (None, None), "PiYG"
    ),
    "w": field_plot_kwargs(
        "w",
        "t_0",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "RdBu_r",
    ),
    "w^2": field_plot_kwargs(
        r"$w^2$",
        "t_0",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "theta": field_plot_kwargs(
        r"$\theta$",
        "t_0",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "RdBu_r",
    ),
    "vertical_heat_flux": field_plot_kwargs(
        r"$w' \theta'$",
        "t_0",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "RdBu",
    ),
    "csDelta": field_plot_kwargs(
        r"$C_s\Delta $",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "cs": field_plot_kwargs(
        r"$C_s$", "t", "z_theta", ("x_centre", "y_centre"), (None, None), "Oranges"
    ),
    "smag_visc_m": field_plot_kwargs(
        r"$k_{momemtum}$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "cs_theta": field_plot_kwargs(
        r"$C_\theta(\Delta) $",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "RdBu",
    ),
    "smag_visc_h": field_plot_kwargs(
        r"$k_{heat}$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "Richardson_dyn": field_plot_kwargs(
        r"$Ri_d$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "Richardson": field_plot_kwargs(
        r"$Ri$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "Richardson_diag": field_plot_kwargs(
        r"$Ri_d$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "s": field_plot_kwargs(
        r"$|S|$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
}

anisotropic_plot_map = {
    "cs_1": field_plot_map["cs"].with_args(label=r"$Cs_1$"),
    "cs_2": field_plot_map["cs"].with_args(label=r"$Cs_2$"),
    "cs_3": field_plot_map["cs"].with_args(label=r"$Cs_3$"),
    "cs_theta_1": field_plot_map["cs"].with_args(label=r"$C{\theta}_1$"),
    "cs_theta_2": field_plot_map["cs"].with_args(label=r"$C{\theta}_2$"),
    "cs_theta_3": field_plot_map["cs"].with_args(label=r"$C{\theta}_3$"),
}

debug_field_plot_map = {
    "s2d": field_plot_kwargs(
        r"$|\langle S \rangle_{2 \Delta}|$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "s4d": field_plot_kwargs(
        r"$|\langle S \rangle_{4 \Delta}|$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "lm": field_plot_kwargs(
        r"$|L_{ij} M_{ij}|$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "mm": field_plot_kwargs(
        r"$|M_{ij} M_{ij}|$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "qn": field_plot_kwargs(
        r"$|Q_{ij} N_{ij}|$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
    "nn": field_plot_kwargs(
        r"$|N_{ij} N_{ij}|$",
        "t",
        "z_theta",
        ("x_centre", "y_centre"),
        (None, None),
        "Oranges",
    ),
}
