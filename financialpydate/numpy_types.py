from datetime import date
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NumpyDateType: TypeAlias = np.datetime64[date | int | None]

DateArrayType: TypeAlias = npt.NDArray[NumpyDateType]

IntArrayType: TypeAlias = npt.NDArray[np.uint64]
