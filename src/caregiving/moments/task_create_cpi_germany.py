"""Get Consumper Price Index for Germany from 1991 to 2020."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from caregiving.config import BLD


def task_process_cpi_data(
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "cpi_germany.csv",
) -> None:
    """Process CPI data."""
    cpi_dict = {
        "int_year": list(range(1991, 2023)),
        "cpi": [
            61.9,
            65.0,
            67.9,
            69.7,
            71.0,
            72.0,
            73.4,
            74.0,
            74.5,
            75.5,
            77.0,
            78.1,
            78.9,
            80.2,
            81.5,
            82.8,
            84.7,
            86.9,
            87.2,
            88.1,
            90.0,
            91.7,
            93.1,
            94.0,
            94.5,
            95.0,
            96.4,
            98.1,
            99.5,
            100.0,
            103.1,
            110.2,
        ],
    }
    cpi = pd.DataFrame(cpi_dict)

    cpi.to_csv(path_to_save)
