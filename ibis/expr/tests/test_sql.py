from __future__ import annotations

import inspect
import itertools
from functools import reduce

import pytest
import sqlglot as sg
from sqlglot.diff import Keep
from sqlglot.optimizer import optimize

import ibis
import ibis.expr.operations.arrays as arr_ops
import ibis.expr.operations.generic as gen_ops
import ibis.expr.operations.logical as logic_ops
import ibis.expr.operations.numeric as num_ops
import ibis.expr.operations.strings as str_ops
import ibis.expr.operations.temporal as time_ops
from ibis.common.temporal import DateUnit, IntervalUnit, TimeUnit
from ibis.expr.sql import Catalog, get_compiler



pytest.importorskip("black")



def parse_param(param):
    def parse_descriptor(descriptor):
        mapper = {
            "numeric": ["float", "int"],
            "integer": ["int"],
            "boolean": ["bool"],
            "string": ["str"],
            "array": ["array<string>"],
            "value": ["float", "int", "bool", "str"],
            "floating": ["float"],
            "date": ["date"],
            "timestamp": ["timestamp"],
            "time": ["time"],
            "temporal": ["timestamp"],
            "interval": ["interval('D')"],
            "dateunit": ["dateunit"],
            "timestampunit": ["timestampunit"],
            "timeunit": ["timeunit"],
            "intervalunit": ["intervalunit"],
        }
        # return mapper.get(descriptor.__name__.lower())
        return mapper[descriptor.__name__.lower()]

    def parse_arg(arg):
        if hasattr(arg, "__args__"):
            if arg.__name__.lower() == "array":
                if isinstance(arg.__args__, tuple):
                    # dtypes = sum(map(parse_descriptor, arg.__args__), [])
                    dtypes = reduce(
                        lambda acc, item: acc + parse_descriptor(item), arg.__args__, []
                    )

                    return [f"array<{s}>" for s in dtypes]

                else:
                    return parse_descriptor(arg)

            elif isinstance(arg.__args__, tuple):
                # return sum(map(parse_descriptor, arg.__args__), [])
                return reduce(
                    lambda acc, item: acc + parse_descriptor(item), arg.__args__, []
                )

            else:
                return parse_descriptor(arg)

        else:
            return parse_descriptor(arg)

    if hasattr(param.annotation.pattern, "params"):
        arg = param.annotation.pattern.params.get("T")

        return parse_arg(arg)

    elif hasattr(param.annotation.pattern, "pattern"):
        arg = param.annotation.pattern.pattern.params.get("T")
        return parse_arg(arg)

    elif hasattr(param.annotation.pattern, "type"):
        arg = param.annotation.pattern.type

        return parse_arg(arg)

    elif hasattr(param.annotation.pattern, "patterns"):
        arg = param.annotation.pattern.patterns[0]
        if isinstance(arg, ibis.common.patterns.SequenceOf):
            dtypes = parse_arg(arg.item.type)
        return dtypes[0], dtypes[0]

    else:
        raise NotImplementedError


def get_arg_types(func):
    sig = inspect.signature(func)

    return {k: parse_param(v) for k, v in sig.parameters.items()}


BACKENDS = [
    "duckdb",
    # "bigquery",
    # "pyspark"
]

not_contains = lambda s, tokens: all(token not in s for token in tokens)
contains = lambda s, tokens: any(token in s for token in tokens)


def array_ops():
    # IMPLEMENTED = ["min", "max", "mean", "size", "all", "any", "unnest"]
    NOT_IMPLEMENTED = [
        "nary",
        "agg",
        "index",
        "range",
        "value",
        "contains",
        "union",
        "slice",
        "flatten",
        "intersect",
        "zip",
        "repeat",
        "distinct",
        "concat",
        "remove",
        "map",
        "position",
        "filter",
    ]
    return filter(
        lambda op: not_contains(op.__name__.lower(), NOT_IMPLEMENTED)
        and op.__name__.lower() != "array",
        [obj for name, obj in inspect.getmembers(arr_ops, inspect.isclass)],
    )


def generic_ops():
    NOT_IMPLEMENTED = [
        "nary",
        "value",
        "random",
        "cast",
        "any",
        "typeof",
        "case",
        "trycast",
        "rowid",
        "hash",
        "scalar",
        "now",
        "instance",
        "impure",
        "hex",
        "nullif",
        # The following functions are implemented elsewhere
        "greatest",
        "coalesce",
        "least",
    ]

    ABSTRACTS = [
        "e",
        "pi",
        "constant",
        "literal",
        "scalar",
        "relation",
        "deferred",
        "typevar",
        "annotated",
        "singleton",
        "impure",
        "length",
    ]
    return filter(
        lambda op: not_contains(op.__name__.lower(), NOT_IMPLEMENTED)
        and op.__name__.lower() not in ABSTRACTS,
        [obj for name, obj in inspect.getmembers(gen_ops, inspect.isclass)],
    )


def numeric_ops():
    NOT_IMPLEMENTED = [
        # NOTE: Abstract ops that shouldn't work
        "nary",
        "value",
        "baseconvert",
        "logarithm",
        "cast",
        "neg",
    ]

    return filter(
        lambda op: not_contains(op.__name__.lower(), NOT_IMPLEMENTED),
        [obj for name, obj in inspect.getmembers(num_ops, inspect.isclass)],
    )


def temporal_ops():
    NOT_IMPLEMENTED = [
        # NOTE: Abstract ops that shouldn't work
        "nary",
        "value",
        "annotated",
        "attrs",
        "unit",
        "coercedto",
        "millisecond",  # duckdb -> EXTRACT('ms' FROM "t0"."in_col0") % 1000
        "microsecond",  # duckdb -> EXTRACT('ms' FROM "t0"."in_col0") % 1000000
        "field",
        "field",
        "dayofweek",  # composed expression
        "timediff",  # OperationNotDefinedError
        "datedelta",  # Not correctly annotated, so testing infrastructure cannot work here
        "timedelta",  # Not correctly annotated, so testing infrastructure cannot work here
        "timestampdelta",  # Not correctly annotated, so testing infrastructure cannot work here
        "temporaldelta",  # OperationNotDefinedError
        "timeadd",  # OperationNotDefinedError
        "timesub",  # OperationNotDefinedError,
        "timestampfromunix",  # # Not correctly annotated, so testing infrastructure cannot work here
        "intervalfloordivide",  # OperationNotDefinedError
        "intervalfrominteger",  # TODO: ANON
        "timestampbucket",  # TODO: complete this function
        "betweentime",  # TODO: complete this function
    ]

    return filter(
        lambda op: not_contains(op.__name__.lower(), NOT_IMPLEMENTED),
        [obj for name, obj in inspect.getmembers(time_ops, inspect.isclass)],
    )


def string_ops():
    """All operations except for the currently unsupported"""
    NOT_IMPLEMENTED = [
        "extract",
        "case",
        "value",
        "array",
        "ascii",
        "stringconcat",  # NOTE: needs to parse the tuple argument in get_arg_types to fix the test. Function may work
        "stringjoin",  # NOTE: needs to parse the tuple argument in get_arg_types to fix the test. Function may work
        # NOTE: the following do not map to a single ibis operation. Maybe a "pipeline conversion" functionality?
        "substring",
        "stringfind",
        "findinset",
        "stringsqlilike",
        "stringsqllike",
        "capitalize",
        "stringslice",  # NOTE: has coalesce and need to implement NULL
    ]

    return filter(
        lambda op: not_contains(
            op.__name__.lower(),
            ["nary", "fuzzy"] + NOT_IMPLEMENTED,
        ),
        [obj for name, obj in inspect.getmembers(str_ops, inspect.isclass)],
    )


def logical_ops():
    """All operations except for the currently unsupported"""
    NOT_IMPLEMENTED = [
        "nary",
        "value",
        "error",
        "comparison",
    ]

    return filter(
        lambda op: not_contains(
            op.__name__.lower(),
            NOT_IMPLEMENTED,
        ),
        [obj for name, obj in inspect.getmembers(logic_ops, inspect.isclass)],
    )


def round_trip(df, expr, backend):
    # This is the source of truth
    sql = ibis.to_sql(expr, dialect=backend)

    katalog = {df.get_name(): df.schema()}
    expr_ = ibis.parse_sql(
        sql,
        catalog=katalog,
        dialect=backend,
    )

    sql_ = ibis.to_sql(expr_, dialect=backend)

    assert asts_are_equal(
        sql,
        sql_,
        Catalog(
            {name: ibis.table(schema, name=name) for name, schema in katalog.items()}
        ).to_sqlglot(),
        dialect=backend,
    ) or expr.equals(expr_)


def asts_are_equal(sql1, sql2, schema, dialect):
    """NOTE: https://github.com/tobymao/sqlglot/blob/main/posts/sql_diff.md ."""
    to_canonical = lambda s: optimize(sg.parse_one(s, dialect=dialect), schema=schema)
    asts = map(to_canonical, [sql1, sql2])
    edit_script = sg.diff(*asts)

    return sum(0 if isinstance(e, Keep) else 1 for e in edit_script) == 0


@pytest.mark.parametrize(
    "op, backend",
    list(itertools.product(numeric_ops(), BACKENDS))
    + list(itertools.product(string_ops(), BACKENDS))
    + list(itertools.product(generic_ops(), BACKENDS))
    + list(itertools.product(array_ops(), BACKENDS))
    + list(itertools.product(logical_ops(), BACKENDS))
    + list(itertools.product(temporal_ops(), BACKENDS)),
    # list(itertools.product([time_ops.TimestampTruncate], BACKENDS))
)
def test_round_trippable(op, backend):
    """
    Asserts that SQL generated from ibis expression graph can be parsed back to the same expression graph.

    NOTE:
        1. compares using sqlglot ast (so that col.not() and not col.isnull() are treated the same)
        2. also compares the ibis expression directly
    """

    # infer the datatypes of the functions and create the catalog
    compiler = get_compiler(backend)
    if op not in compiler.UNSUPPORTED_OPS:
        sig = get_arg_types(op)

        df = ibis.table(
            {k: v[0] for k, v in sig.items() if not_contains(v[0], ["unit"])},
            name="tmp_table",
        )

        params = dict()
        for k, v in sig.items():
            if v[0] == "dateunit":
                params[k] = DateUnit.DAY.value
            elif v[0] == "timeunit":
                params[k] = TimeUnit.HOUR.value
            elif v[0] == "intervalunit":
                params[k] = IntervalUnit.DAY.value
            # elif v[0].startswith("interval("):
            #     params[k] = ibis.interval(1, unit='D')

        expr = df.mutate(
            out_col=op(**{k: df[k] for k in sig if k not in params}, **params).to_expr()
        )

        round_trip(df, expr, backend)


@pytest.mark.parametrize(
    "op, backend",
    list(
        itertools.product([gen_ops.Greatest, gen_ops.Least, gen_ops.Coalesce], BACKENDS)
    ),
)
def test_round_trippable_with_array_args(op, backend):
    """
    Asserts that SQL generated from ibis expression graph can be parsed back to the same expression graph.

    NOTE:
        1. compares using sqlglot ast (so that col.not() and not col.isnull() are treated the same)
        2. also compares the ibis expression directly
    """

    # infer the datatypes of the functions and create the catalog
    compiler = get_compiler(backend)
    if op not in compiler.UNSUPPORTED_OPS:
        sig = get_arg_types(op)
        cols = dict()
        for name, dtypes in sig.items():
            if isinstance(dtypes, tuple):
                for k in range(len(dtypes)):
                    cols[f"{name}_{k}"] = dtypes[k]
        df = ibis.table(cols, name="tmp_table")

        expr = df.mutate(out_col=op(tuple([df[k] for k in cols])).to_expr())
        round_trip(df, expr, backend)

catalog = {
    "employee": {"first_name": "string", "last_name": "string", "id": "int64"},
    "call": {
        "start_time": "timestamp",
        "end_time": "timestamp",
        "employee_id": "int64",
        "call_outcome_id": "int64",
        "call_attempts": "int64",
    },
    "call_outcome": {"outcome_text": "string", "id": "int64"},
}


def test_parse_sql_basic_projection(snapshot):
    sql = "SELECT *, first_name as first FROM employee WHERE id < 5 ORDER BY id DESC"
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


@pytest.mark.parametrize("how", ["right", "left", "inner"])
def test_parse_sql_basic_join(how, snapshot):
    sql = f"""
SELECT
  *,
  first_name as first
FROM employee {how.upper()}
JOIN call ON
  employee.id = call.employee_id
WHERE
  id < 5
ORDER BY
  id DESC"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_multiple_joins(snapshot):
    sql = """
SELECT *
FROM employee
JOIN call
  ON employee.id = call.employee_id
JOIN call_outcome
  ON call.call_outcome_id = call_outcome.id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_basic_aggregation(snapshot):
    sql = """
SELECT
  employee_id,
  sum(call_attempts) AS attempts
FROM call
GROUP BY employee_id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_basic_aggregation_with_join(snapshot):
    sql = """
SELECT
  id,
  sum(call_attempts) AS attempts
FROM employee
LEFT JOIN call
  ON employee.id = call.employee_id
GROUP BY id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_aggregation_with_multiple_joins(snapshot):
    sql = """
SELECT
  t.employee_id,
  AVG(t.call_attempts) AS avg_attempts
FROM (
  SELECT * FROM employee JOIN call ON employee.id = call.employee_id
  JOIN call_outcome ON call.call_outcome_id = call_outcome.id
) AS t
GROUP BY t.employee_id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_simple_reduction(snapshot):
    sql = """SELECT AVG(call_attempts) AS mean FROM call"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_scalar_subquery(snapshot):
    sql = """
SELECT *
FROM call
WHERE call_attempts > (
  SELECT avg(call_attempts) AS mean
  FROM call
)"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_simple_select_count(snapshot):
    sql = """SELECT COUNT(first_name) FROM employee"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_table_alias(snapshot):
    sql = """SELECT e.* FROM employee AS e"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_join_with_filter(snapshot):
    sql = """
SELECT *, first_name as first FROM employee
LEFT JOIN call ON employee.id = call.employee_id
WHERE id < 5
ORDER BY id DESC"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_in_clause(snapshot):
    sql = """
SELECT first_name FROM employee
WHERE first_name IN ('Graham', 'John', 'Terry', 'Eric', 'Michael')"""

    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")
