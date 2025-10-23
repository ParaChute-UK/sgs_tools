from pathlib import Path


def build_output_fname(base: str | Path, *tags: str, ext: str = ".nc") -> Path:
    """
    Constructs a standardized output filename by '_'.join a base name and optional ordered list of tags.

    :param base: The core name of the file, can include directory structure but better not.
    :param tags: Additional string components to include in the filename (e.g., script tag, version, context).
                Any empty tags will be discarded, everything else is turned to string via str().
    :param ext: The file extension to use.
    :return: A Path object representing the constructed filename.
    """

    # Treat as directory if it is an existing directory,
    # or if the string ends with a separator
    base = Path(base)
    looks_like_dir = base.is_dir() or str(base).endswith(("/", "\\"))

    if looks_like_dir:
        parent_dir = base
        fname = ""
    else:
        parent_dir = base.parent
        fname = base.stem

    parts = [fname] + list(tags)
    parts = [str(a) for a in parts if a]
    assert parts, "Can't make a filename from empty strings"
    fullname = parent_dir / "_".join(parts)
    return fullname.with_suffix(ext)
