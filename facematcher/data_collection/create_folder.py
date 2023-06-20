# MANAGE ENVIRONNEMENT
from pathlib import Path
import os


def create_folder(dirpath_name: str, subdir_names: list) -> None:
    """create directories and sub-directories for data"""

    dirpath = Path(os.path.join(dirpath_name))

    if dirpath.is_dir():
        print()
        print(f"The directory '{dirpath_name}' already exists")

        # Check and create the subdirectory
        for subdir_name in subdir_names:

            subdir = Path(os.path.join(subdir_name))
            subpath = dirpath / subdir
            if subpath.is_dir():
                print(f"The sub-directories '{subdir_name}' already exists")
            else:
                subpath.mkdir()
                print(f"The sub-directories '{subdir_name}' has been created")
    else:
        # Create the directory
        dirpath.mkdir()

        # Create subdirectories
        for subdir_name in subdir_names:

            subdir = Path(os.path.join(subdir_name))
            subpath = dirpath / subdir
            subpath.mkdir()

        subdir_names_str = ', '.join(f"'{subdir_name}'"
                                     for subdir_name in subdir_names)
        print()
        print(
            f"The directory '{dirpath_name}' and sub-directories {subdir_names_str} "
            "have been created"
            )
        print()
