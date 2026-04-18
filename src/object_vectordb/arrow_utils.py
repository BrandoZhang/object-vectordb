"""pyarrow helpers: Python value -> Arrow type inference, record-batch construction.

Kept in a separate module so that `store.py` stays free of pyarrow imports.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from .exceptions import SchemaError


def python_value_to_arrow_type(value: Any) -> pa.DataType:
    """Infer a pyarrow DataType from a single non-None Python sample value.

    Rules (conservative):
      bool     -> bool   (must come BEFORE int: bool is a subclass of int)
      int      -> int64
      float    -> float64
      str      -> string
      bytes    -> binary
      list[x]  -> list_<element type inferred from first non-None item>
      dict     -> string (JSON-encoded; struct support deferred)

    Raises SchemaError on a None sample or a list whose elements are all None.
    """
    if value is None:
        raise SchemaError(
            "Cannot infer column type from a None sample value. "
            "Write a non-None value first, or add the column with a typed schema."
        )
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, str):
        return pa.string()
    if isinstance(value, bytes):
        return pa.binary()
    if isinstance(value, list):
        for item in value:
            if item is not None:
                return pa.list_(python_value_to_arrow_type(item))
        raise SchemaError("Cannot infer list element type: all items are None.")
    if isinstance(value, dict):
        return pa.string()
    raise SchemaError(f"Unsupported property value type: {type(value).__name__}")


def encode_property_value(value: Any, arrow_type: pa.DataType) -> Any:
    """Coerce a Python value to the representation expected by `arrow_type`.

    JSON-encodes dicts when the column type is string.
    """
    if value is None:
        return None
    if pa.types.is_string(arrow_type) and isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def arrow_type_to_sql_type(arrow_type: pa.DataType) -> str:
    """Map an Arrow type to a SQL type string usable in DataFusion CAST expressions.

    Used for null-clearing scalar property cells via table.update(values_sql=...).
    """
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    if pa.types.is_integer(arrow_type):
        return "BIGINT"
    if pa.types.is_floating(arrow_type):
        return "DOUBLE"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "STRING"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "BINARY"
    # For list / struct / fixed_size_list we don't use values_sql null-clearing
    # (vector null-clear goes through merge_insert instead).
    raise SchemaError(
        f"No SQL null-cast available for arrow type {arrow_type}; "
        "use merge_insert with a typed null batch instead."
    )
