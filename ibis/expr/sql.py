from __future__ import annotations

import contextlib
import operator
from functools import partial, singledispatch

import sqlglot as sg
import sqlglot.expressions as sge
import sqlglot.optimizer as sgo
import sqlglot.planner as sgp
from public import public

import ibis
import ibis.common
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.sql import compilers
from ibis.backends.sql.compilers.base import SQLGlotCompiler
from ibis.backends.sql.datatypes import SqlglotType
from ibis.common.temporal import DateUnit, IntervalUnit
from ibis.expr.operations.core import Value
from ibis.util import experimental


class Catalog(dict[str, sch.Schema]):
    """A catalog of tables and their schemas."""

    typemap = {
        dt.Int8: "tinyint",
        dt.Int16: "smallint",
        dt.Int32: "int",
        dt.Int64: "bigint",
        dt.Float16: "halffloat",
        dt.Float32: "float",
        dt.Float64: "double",
        dt.Decimal: "decimal",
        dt.Boolean: "boolean",
        dt.JSON: "json",
        dt.Interval: "interval",
        dt.Timestamp: "datetime",
        dt.Date: "date",
        dt.Binary: "varbinary",
        dt.String: "varchar",
        dt.Array: "array",
        dt.Map: "map",
        dt.UUID: "uuid",
        dt.Struct: "struct",
    }

    def to_sqlglot_dtype(self, dtype: dt.DataType) -> str:
        if dtype.is_geospatial():
            return dtype.geotype
        else:
            default = dtype.__class__.__name__.lower()
            return self.typemap.get(type(dtype), default)

    def to_sqlglot_schema(self, schema: sch.Schema) -> dict[str, str]:
        return {name: self.to_sqlglot_dtype(dtype) for name, dtype in schema.items()}

    def to_sqlglot(self):
        return {
            name: self.to_sqlglot_schema(table.schema()) for name, table in self.items()
        }

    def overlay(self, step, compiler):
        updates = {
            dep.name: convert(dep, catalog=self, compiler=compiler)
            for dep in step.dependencies
        }

        # handle scan aliases: FROM foo AS bar
        source = getattr(step, "source", None)
        alias = getattr(source, "args", {}).get("alias")
        if alias is not None and (source_name := self.get(source.name)) is not None:
            self[alias.name] = source_name

        return Catalog({**self, **updates})


@singledispatch
def convert(step, catalog, compiler):
    raise TypeError(type(step))


# @convert.register(sgp.Aggregate)
# def convert_aggregate(agg, catalog, compiler):
#     catalog = catalog.overlay(agg, compiler)
#     source = catalog[agg.source]

#     if agg.aggregations:
#         metrics = [
#             convert(metric, catalog=catalog, compiler=compiler)
#             for metric in agg.aggregations
#         ]

#         groups = [
#             convert(g, catalog=catalog, compiler=compiler) for k, g in agg.group.items()
#         ]
#         if groups:
#             source = source.aggregate(metrics, by=groups)
#         else:
#             source = source.mutate(*metrics)

#     return source


@convert.register(sgp.Scan)
def convert_scan(scan, catalog, compiler):
    catalog = catalog.overlay(scan, compiler)
    table = catalog[scan.source.alias_or_name]

    if scan.condition:
        pred = convert(scan.condition, catalog=catalog, compiler=compiler)
        table = table.filter(pred)

    if scan.projections:
        projs = [
            convert(proj, catalog=catalog, compiler=compiler)
            for proj in scan.projections
        ]
        table = table.select(projs)

    if isinstance(scan.limit, int):
        table = table.limit(scan.limit)

    return table


def qualify_projections(projections, groups):
    # The sqlglot planner will (sometimes) alias projections to the aggregate
    # that precedes it.
    #
    # - Sort: lineitem (132849388268768)
    #   Context:
    #     Key:
    #       - "l_returnflag"
    #       - "l_linestatus"
    #   Projections:
    #     - lineitem._g0 AS "l_returnflag"
    #     - lineitem._g1 AS "l_linestatus"
    #     <snip>
    #   Dependencies:
    #   - Aggregate: lineitem (132849388268864)
    #     Context:
    #       Aggregations:
    #         <snip>
    #       Group:
    #         - "lineitem"."l_returnflag"  <-- this is _g0
    #         - "lineitem"."l_linestatus"  <-- this is _g1
    #         <snip>
    #
    #  These aliases are stored in a dictionary in the aggregate `groups`, so if
    #  those are pulled out beforehand then we can use them to replace the
    #  aliases in the projections.

    def transformer(node):
        if isinstance(node, sge.Alias) and (name := node.this.name).startswith("_g"):
            return groups[name]
        return node

    projects = [project.transform(transformer) for project in projections]

    return projects


@convert.register(sgp.Sort)
def convert_sort(sort, catalog, compiler):
    catalog = catalog.overlay(sort, compiler)

    table = catalog[sort.name]

    if sort.key:
        keys = [convert(key, catalog=catalog, compiler=compiler) for key in sort.key]
        table = table.order_by(keys)

    if sort.projections:
        groups = {}
        # group definitions that may be used in projections are defined
        # in the aggregate in dependencies...
        for dep in sort.dependencies:
            if (group := getattr(dep, "group", None)) is not None:
                groups |= group
        projs = [
            convert(proj, catalog=catalog, compiler=compiler)
            for proj in qualify_projections(sort.projections, groups)
        ]
        table = table.select(projs)

    if isinstance(sort.limit, int):
        table = table.limit(sort.limit)

    return table


_join_types = {
    "": "inner",
    "LEFT": "left",
    "RIGHT": "right",
}


@convert.register(sgp.Join)
def convert_join(join, catalog, compiler):
    catalog = catalog.overlay(join, compiler)

    left_name = join.name
    left_table = catalog[left_name]

    for right_name, desc in join.joins.items():
        right_table = catalog[right_name]
        join_kind = _join_types[desc["side"]]

        predicate = None
        if desc["join_key"]:
            for left_key, right_key in zip(desc["source_key"], desc["join_key"]):
                left_key = convert(left_key, catalog=catalog, compiler=compiler)
                right_key = convert(right_key, catalog=catalog, compiler=compiler)
                if predicate is None:
                    predicate = left_key == right_key
                else:
                    predicate &= left_key == right_key

        if "condition" in desc.keys():
            condition = desc["condition"]
            if predicate is None:
                predicate = convert(condition, catalog=catalog, compiler=compiler)
            else:
                predicate &= convert(condition, catalog=catalog, compiler=compiler)

        left_table = left_table.join(right_table, predicates=predicate, how=join_kind)

    if join.condition:
        predicate = convert(join.condition, catalog=catalog, compiler=compiler)
        left_table = left_table.filter(predicate)

    catalog[left_name] = left_table

    return left_table


def replace_operands(agg):
    # The sqlglot planner will pull out computed operands into a separate
    # section and alias them #
    # e.g.
    # Context:
    #   Aggregations:
    #     - SUM("_a_0") AS "sum_disc_price"
    #   Operands:
    #     - "lineitem"."l_extendedprice" * (1 - "lineitem"."l_discount") AS _a_0
    #
    # For the purposes of decompiling, we want these to be inline, so here we
    # replace those new aliases with the parsed sqlglot expression
    operands = {operand.alias: operand.this for operand in agg.operands}

    def transformer(node):
        if isinstance(node, sge.Column) and node.name in operands.keys():
            return operands[node.name]
        return node

    aggs = [item.transform(transformer) for item in agg.aggregations]

    agg.aggregations = aggs

    return agg


@convert.register(sgp.Aggregate)
def convert_aggregate(agg, catalog, compiler):
    catalog = catalog.overlay(agg, compiler=compiler)

    agg = replace_operands(agg)

    table = catalog[agg.source]
    if agg.aggregations:
        metrics = [
            convert(a, catalog=catalog, compiler=compiler) for a in agg.aggregations
        ]
        groups = [
            convert(g, catalog=catalog, compiler=compiler) for k, g in agg.group.items()
        ]
        table = table.aggregate(metrics, by=groups)

    return table


@convert.register(sge.Subquery)
def convert_subquery(subquery, catalog, compiler):
    # Check if this is an UNNEST subquery
    if isinstance(subquery.this.args["from"].this, sge.Unnest):
        return convert_unnest_collection_as_subquery(subquery, catalog, compiler)

    # If not an UNNEST subquery, proceed with the regular optimization
    tree = sgo.optimize(subquery, catalog.to_sqlglot(), rules=sgo.RULES)
    plan = sgp.Plan(tree)
    return convert(plan.root, catalog=catalog, compiler=compiler)


def convert_unnest_collection_as_subquery(subquery, catalog, compiler):
    # NOTE: array method names end with s...
    agg_fun = subquery.this.expressions[0].this
    col = convert(
        sge.convert(subquery.this.args["from"].this.expressions[0]), catalog, compiler
    )

    reduction_method = _reduction_methods[type(agg_fun)]

    if not reduction_method.endswith("s"):
        reduction_method += "s"
    return getattr(col, reduction_method)()


@convert.register(sge.Literal)
def convert_literal(literal, catalog, compiler):
    value = literal.this
    if literal.is_int:
        value = int(value)
    elif literal.is_number:
        value = float(value)

    return ibis.literal(value)


@convert.register(sge.Boolean)
def convert_boolean(boolean, catalog, compiler):
    return ibis.literal(boolean.this)


@convert.register(sge.If)
def convert_if(st, catalog, compiler):
    params = {"bool_expr": "this", "true_expr": "true", "false_null_expr": "false"}

    params = {
        k: maybe_convert(st.args.get(v), catalog, compiler) for k, v in params.items()
    }
    return ops.IfElse(**params).to_expr()


@convert.register(sge.Case)
def convert_case(statement, catalog, compiler):
    base = statement.args.get("this")
    base = maybe_convert(base, catalog, compiler)

    def convert_if_(k, catalog, compiler):
        """TODO: confirm my understanding with the Ibis Team."""
        params = {"bool_expr": "this", "true_expr": "true", "false_null_expr": "false"}
        case_ = maybe_convert(k.args.get(params.get("bool_expr")), catalog, compiler)
        result = maybe_convert(k.args.get(params.get("true_expr")), catalog, compiler)

        return case_, result

    ifs = [convert_if_(k, catalog, compiler) for k in statement.args.get("ifs")]

    default = convert(statement.args.get("default"), catalog=catalog, compiler=compiler)
    cases = [tu[0] for tu in ifs]
    results = [tu[1] for tu in ifs]

    return ops.SearchedCase(
        cases=cases,
        results=results,
        default=default,
    ).to_expr()


@convert.register(sge.Dot)
def convert_dot(dot, catalog, compiler):
    this = convert(dot.expression, catalog=catalog, compiler=compiler)
    return this


@convert.register(sge.Alias)
def convert_alias(alias, catalog, compiler):
    this = convert(alias.this, catalog=catalog, compiler=compiler)

    if isinstance(this, Value):
        return this.to_expr().name(alias.alias_or_name)

    elif isinstance(this, ibis.expr.types.relations.Table):
        return this.alias(alias.alias_or_name)

    else:
        return this.name(alias.alias_or_name)


@convert.register(sge.Column)
def convert_column(column, catalog, compiler):
    table = catalog[column.table]

    return table[column.name]


@convert.register(sge.Null)
def convert_null(expr, catalog, compiler):
    return ops.NULL


@convert.register(sge.Ordered)
def convert_ordered(ordered, catalog, compiler):
    this = ibis._[ordered.this.name]
    desc = ordered.args.get("desc", False)  # not exposed as an attribute
    nulls_first = ordered.args.get("nulls_first", False)
    return (
        ibis.desc(this, nulls_first=nulls_first)
        if desc
        else ibis.asc(this, nulls_first=nulls_first)
    )


_unary_operations = {
    sge.Paren: lambda x: x,
    sge.Not: ops.Not,
    sge.Neg: ops.Negate,
    sge.BitwiseNot: ops.BitwiseNot,
}


@convert.register(sge.Unary)
def convert_unary(unary, catalog, compiler):
    op = _unary_operations[type(unary)]
    this = convert(unary.this, catalog=catalog, compiler=compiler)
    return op(this)


_binary_operations = {
    sge.LT: operator.lt,
    sge.LTE: operator.le,
    sge.GT: operator.gt,
    sge.GTE: operator.ge,
    sge.EQ: operator.eq,
    sge.NEQ: operator.ne,
    sge.Add: operator.add,
    sge.Sub: operator.sub,
    sge.Mul: operator.mul,
    sge.Div: operator.truediv,
    sge.Pow: operator.pow,
    sge.And: operator.and_,
    sge.Or: operator.or_,
    sge.Mod: operator.mod,
    sge.DPipe: operator.concat,
    sge.RegexpLike: ops.RegexSearch,
    sge.NullSafeEQ: ops.IdenticalTo,
    sge.BitwiseXor: ops.BitwiseXor,
    sge.BitwiseOr: ops.BitwiseOr,
    sge.BitwiseAnd: ops.BitwiseAnd,
    sge.BitwiseRightShift: ops.BitwiseRightShift,
    sge.BitwiseLeftShift: ops.BitwiseLeftShift,
}


@convert.register(sge.Binary)
def convert_binary(binary, catalog, compiler):
    this = convert(binary.this, catalog=catalog, compiler=compiler)
    expr = convert(binary.expression, catalog=catalog, compiler=compiler)

    if isinstance(binary, sge.Is):
        if expr == ops.NULL:
            return ops.IsNull(this).to_expr()
    else:
        op = _binary_operations[type(binary)]

    if isinstance(binary.expression, sge.Subquery):
        # expr is a table expression
        assert len(expr.columns) == 1
        name = expr.columns[0]
        expr = expr[name]

    as_expr = lambda obj: (
        obj.to_expr()
        if not any(
            isinstance(this, dtype)
            for dtype in [ibis.expr.types.Column, ibis.expr.types.Scalar]
        )
        else obj
    )
    this = as_expr(this)
    expr = as_expr(expr)

    if isinstance(op, Value):
        return op(this, expr).to_expr()

    else:
        return op(this, expr)


_reduction_methods = {
    sge.Max: "max",
    sge.Min: "min",
    sge.Quantile: "quantile",
    sge.Sum: "sum",
    sge.Avg: "mean",
    sge.Count: "count",
    sge.LogicalAnd: "all",
    sge.LogicalOr: "any",
}


@convert.register(sge.AggFunc)
def convert_sum(reduction, catalog, compiler):
    method = _reduction_methods[type(reduction)]
    this = convert(reduction.this, catalog=catalog, compiler=compiler)
    return getattr(this, method)()


@convert.register(sge.In)
def convert_in(in_, catalog, compiler):
    this = convert(in_.this, catalog=catalog, compiler=compiler)
    candidates = [
        convert(expression, catalog, compiler) for expression in in_.expressions
    ]
    return this.isin(candidates)


@convert.register(sge.Cast)
def cast(cast, catalog, compiler):
    this = convert(cast.this, catalog, compiler)
    to = convert(cast.to, catalog, compiler)

    return this.cast(to)


@convert.register(sge.DataType)
def datatype(datatype, catalog, compiler):
    return SqlglotType().to_ibis(datatype)


@convert.register(sge.Count)
def count(count, catalog, compiler):
    return ibis._.count()


@public
@experimental
def parse_sql(sqlstring, catalog, dialect=None):
    """Parse a SQL string into an Ibis expression.

    Parameters
    ----------
    sqlstring : str
        SQL string to parse
    catalog : dict
        A dictionary mapping table names to either schemas or ibis table expressions.
        If a schema is passed, a table expression will be created using the schema.
    dialect : str, optional
        The SQL dialect to use with sqlglot to parse the query string.

    Returns
    -------
    expr : ir.Expr

    """
    catalog = Catalog(
        {name: ibis.table(schema, name=name) for name, schema in catalog.items()}
    )

    expr = sg.parse_one(sqlstring, dialect)
    tree = sgo.optimize(expr, catalog.to_sqlglot(), rules=sgo.RULES)
    plan = sgp.Plan(tree)

    compiler = get_compiler(dialect)()

    return convert(plan.root, catalog=catalog, compiler=compiler)


class SQLString(str):
    """Object to hold a formatted SQL string.

    Syntax highlights in Jupyter notebooks.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)!r})"

    def _repr_markdown_(self) -> str:
        return f"```sql\n{self!s}\n```"

    def _repr_pretty_(self, p, cycle) -> str:
        output = str(self)
        try:
            from pygments import highlight
            from pygments.formatters import TerminalFormatter
            from pygments.lexers import SqlLexer
        except ImportError:
            pass
        else:
            with contextlib.suppress(Exception):
                output = highlight(
                    code=output,
                    lexer=SqlLexer(),
                    formatter=TerminalFormatter(),
                )

        # strip trailing newline
        p.text(output.strip())


@convert.register(sge.Window)
def convert_window(window, catalog, compiler):
    operands = window.unnest_operands()
    params = dict()

    bounds = {"preceding": "start", "following": "end"}

    # Parse bounds
    for name, token in bounds.items():
        if operands[-1].args.get(f"{token}_side", "").lower() == name:
            obj = operands[-1].args.get(token)
            params[name] = (
                convert(obj, catalog, compiler) if obj != "UNBOUNDED" else None
            )

        else:
            params[name] = 0

    if isinstance(operands[1], sge.Column):
        params["group_by"] = operands[1].this.this

    if isinstance(operands[-2], sge.Order):
        params["order_by"] = list(
            map(
                partial(convert, catalog=catalog, compiler=compiler),
                operands[-2].args.get("expressions"),
            )
        )

    kind = operands[-1].args.get("kind", "").upper()

    if all(params.get(name) is None for name in bounds):
        # NOTE: don't know why but have to call ibis.window
        w = ibis.window

    elif kind == "ROWS":
        w = ibis.rows_window
    elif kind == "RANGE":
        w = ibis.range_window

    else:
        raise NotImplementedError

    return convert(operands[0], catalog, compiler).over(w(**params))


@convert.register(sge.Log)
def convert_log(func, catalog, compiler):
    if func.expression is None:
        col = convert(func.this, catalog=catalog, compiler=compiler)
        res = col.log10()
    else:
        col = convert(func.expression, catalog=catalog, compiler=compiler)
        base = convert(func.this, catalog=catalog, compiler=compiler)

        if not isinstance(base, ibis.expr.types.Column):
            if str(base) == str(2):
                res = col.log2()

            elif str(base) == str(10):
                res = col.log10()

            else:
                raise NotImplementedError

        else:
            res = col.log(base)

    return res


def maybe_convert(node, catalog, compiler):
    if isinstance(node, list):
        return [maybe_convert(n, catalog, compiler) for n in node]
    if node is not None:
        return convert(node, catalog, compiler)
    else:
        return None


@convert.register(sge.Coalesce)
def convert_coalesce(func, catalog, compiler):
    this = convert(func.this, catalog=catalog, compiler=compiler)
    params = maybe_convert(func.args.get("expressions"), catalog, compiler)
    op = ops.Coalesce
    return op(tuple([this] + params)).to_expr()


@convert.register(sge.Least)
def convert_least(func, catalog, compiler):
    this = convert(func.this, catalog=catalog, compiler=compiler)
    params = maybe_convert(func.args.get("expressions"), catalog, compiler)
    op = ops.Least
    return op(tuple([this] + params)).to_expr()


@convert.register(sge.Greatest)
def convert_greatest(func, catalog, compiler):
    this = convert(func.this, catalog=catalog, compiler=compiler)
    params = maybe_convert(func.args.get("expressions"), catalog, compiler)
    op = ops.Greatest
    return op(tuple([this] + params)).to_expr()


@convert.register(sge.Substring)
def convert_substr(func, catalog, compiler):
    this = convert(func.this, catalog=catalog, compiler=compiler)
    keys = [k for k in func.arg_types if k not in ["this", "is_left"]]
    sg_params = {k: maybe_convert(func.args.get(k), catalog, compiler) for k in keys}
    sg_params["start"] -= 1

    op = ops.Substring
    return op(this, **sg_params).to_expr()


@convert.register(sge.Round)
def convert_round(func, catalog, compiler):
    this = convert(func.this, catalog=catalog, compiler=compiler)
    decimals = maybe_convert(func.args.get("decimals"), catalog, compiler)
    if decimals is not None:
        op = ops.Round(this, digits=decimals)
    else:
        op = ops.Floor(this)

    return op.to_expr()


@convert.register(sge.Pad)
def convert_pad(func, catalog, compiler):
    this = convert(func.this, catalog=catalog, compiler=compiler)
    keys = [k for k in func.arg_types if k not in ["this", "is_left"]]
    sg_params = {k: maybe_convert(func.args.get(k), catalog, compiler) for k in keys}

    if func.args.get("is_left"):
        op = ops.LPad
    else:
        op = ops.RPad

    return op(
        this, pad=sg_params.get("fill_pattern"), length=sg_params.get("expression")
    ).to_expr()


@convert.register(sge.Var)
def convert_var(func, catalog, compiler):
    return IntervalUnit.from_string(func.this).value


@convert.register(sge.DateSub)
def convert_datesub(func, catalog, compiler):
    return convert_interval_op(func, catalog, compiler, ops.DateSub)


@convert.register(sge.DateAdd)
def convert_dateadd(func, catalog, compiler):
    return convert_interval_op(func, catalog, compiler, ops.DateAdd)


def convert_interval_op(func, catalog, compiler, op):
    this = convert(func.this, catalog, compiler)

    value = int(func.expression.this)
    unit = func.unit.this
    return op(this, ibis.interval(value, unit=unit[0])).to_expr()


@convert.register(sge.Extract)
def convert_extract(func, catalog, compiler):
    this = str(func.this.this)
    col = convert(func.expression, catalog, compiler)

    mapper = {
        k.upper(): k
        for k in [
            "EpochSeconds",
            "Year",
            "Month",
            "Day",
            "DayOfYear",
            "Quarter",
            "WeekOfYear",
            "Hour",
            "Minute",
            "Second",
            "Millisecond",
            "TimeField" "TemporalField",
        ]
    }
    mapper["ms"] = "Millisecond"
    mapper["us"] = "Microsecond"
    mapper["WEEK"] = "WeekOfYear"
    mapper["ISOWEEK"] = "WeekOfYear"
    mapper["ISOYEAR"] = "IsoYear"

    method = getattr(ops, f"Extract{mapper.get(this)}")

    return method(col).to_expr()


@convert.register(sge.DatetimeDiff)
def convert_datetimediff(func, catalog, compiler):
    op = ops.TimestampDelta
    part = convert(func.unit, catalog, compiler)
    left = convert(func.this, catalog, compiler)
    right = convert(func.expression, catalog, compiler)

    return op(part, left, right).to_expr()


@convert.register(sge.DateTrunc)
def convert_datetrunc(func, catalog, compiler):
    op = ops.DateTruncate
    unit = DateUnit.from_string(func.unit.this).value
    arg = convert(func.this, catalog, compiler)

    return op(arg, unit).to_expr()


@convert.register(sge.TimestampTrunc)
def convert_timestamptrunc(func, catalog, compiler):
    unit = IntervalUnit.from_string(func.unit.this).value
    arg = convert(func.this, catalog, compiler)

    op = ops.TimestampTruncate

    # TODO: overwrite rules from ibis compiler...
    if "date" in type(arg).__name__.lower():
        op = ops.DateTruncate
    elif "timecolumn" in type(arg).__name__.lower():
        op = ops.TimeTruncate

    return op(arg, unit).to_expr()


@convert.register(sge.DateFromParts)
def convert_datefromparts(func, catalog, compiler):
    params = ["year", "month", "day"]
    params = {k: convert(func.args.get(k), catalog, compiler) for k in params}

    return ops.DateFromYMD(**params).to_expr()


@convert.register(sge.TimeFromParts)
def convert_timefromparts(func, catalog, compiler):
    params = {"hours": "hour", "minutes": "min", "seconds": "sec"}
    params = {
        k: convert(func.args.get(v), catalog, compiler) for k, v in params.items()
    }

    return ops.TimeFromHMS(**params).to_expr()


# @convert.register(sge.TsOrDsToTimestamp)
# def convert_ts_or_ds_to_timestamp(func, catalog, compiler):
#     this = convert(func.this, catalog, compiler)
#     raise ValueError("Parser lost format string in the translation")


@convert.register(sge.TimestampFromParts)
def convert_timestampfromparts(func, catalog, compiler):
    params = {
        "year": "year",
        "month": "month",
        "day": "day",
        "hours": "hour",
        "minutes": "min",
        "seconds": "sec",
    }
    params = {
        k: convert(func.args.get(v), catalog, compiler) for k, v in params.items()
    }

    return ops.TimestampFromYMDHMS(**params).to_expr()


@convert.register(sge.Func)
def convert_func(func, catalog, compiler):
    this = convert(func.this, catalog=catalog, compiler=compiler)

    skip_params = ["this", "is_left", "binary"]
    keys = [k for k in func.arg_types if k not in skip_params]
    sg_params = {
        k: maybe_convert(func.args.get(k), catalog, compiler)
        for k in keys
        if k not in skip_params
    }

    # NOTE: mapper object points to the correspondent operation and maps the arguments
    # TODO: discuss this approach with Ibis team

    this = convert(func.this, catalog=catalog, compiler=compiler)

    op_name = type(func)

    mapper = {
        sge.RegexpExtract: dict(
            OP_NAME=ops.RegexExtract,
            pattern="expression",
            index="group",
        ),
        sge.RegexpReplace: dict(
            OP_NAME=ops.RegexReplace, pattern="expression", replacement="replacement"
        ),
        sge.RegexpSplit: dict(OP_NAME=ops.RegexSplit, pattern="expression"),
        sge.RegexpLike: dict(OP_NAME=ops.RegexSearch, pattern="expression"),
        sge.Ln: dict(OP_NAME=ops.Ln),
        sge.Trim: dict(OP_NAME=ops.Strip),
        sge.Split: dict(OP_NAME=ops.StringSplit, delimiter="expression"),
        sge.Repeat: dict(OP_NAME=ops.Repeat, times="times"),
        sge.StartsWith: dict(OP_NAME=ops.StartsWith, start="expression"),
        sge.Levenshtein: dict(OP_NAME=ops.Levenshtein, right="expression"),
        sge.Length: dict(
            OP_NAME=ops.StringLength,
        ),
        sge.Right: dict(OP_NAME=ops.StrRight, nchars="expression"),
        sge.Explode: dict(OP_NAME=ops.Unnest),
        sge.ArraySize: dict(OP_NAME=ops.ArrayLength),
        sge.ArraySum: dict(OP_NAME=ops.ArraySum),
        sge.SortArray: dict(OP_NAME=ops.ArraySort),
        sge.Abs: dict(OP_NAME=ops.Abs),
        sge.Sign: dict(OP_NAME=ops.Sign),
        sge.IsInf: dict(OP_NAME=ops.IsInf),
        sge.Sqrt: dict(OP_NAME=ops.Sqrt),
        sge.Exp: dict(OP_NAME=ops.Exp),
        sge.Ceil: dict(OP_NAME=ops.Ceil),
        sge.Ceil: dict(OP_NAME=ops.Ceil),
        sge.IsNan: dict(OP_NAME=ops.IsNan),
        sge.Floor: dict(OP_NAME=ops.Floor),
        sge.TimeToUnix: dict(OP_NAME=ops.ExtractEpochSeconds),
        sge.TimeToStr: dict(OP_NAME=ops.Strftime, format_str="format"),
        sge.DayOfWeek: dict(OP_NAME=ops.DayOfWeekName),
        sge.StrToTime: dict(OP_NAME=ops.StringToTimestamp, format_str="format"),
        sge.UnixToTime: dict(OP_NAME=partial(ops.TimestampFromUNIX, unit="s")),
        sge.StrToDate: dict(OP_NAME=ops.StringToDate, format_str="format"),
        sge.TimeTrunc: dict(OP_NAME=ops.TimeTruncate, unit="unit"),
    }

    op = mapper[op_name].get("OP_NAME")
    param_mapper = mapper[op_name]

    params = {k: sg_params.get(v) for k, v in param_mapper.items() if k != "OP_NAME"}

    return op(this, **params).to_expr()


def get_compiler(dialect):
    # Use a generator expression with next to find the first matching compiler name
    compiler_name = next(
        (
            name
            for name in dir(compilers)
            if dialect in name.lower() and name.endswith("Compiler")
        ),
        None,
    )

    if compiler_name is None:
        raise ValueError(f"No compiler found for dialect: {dialect}")

    return getattr(compilers, compiler_name)


@convert.register(sge.Array)
def convert_array(array, catalog, compiler):
    if isinstance(array.expressions[0], sge.Select):
        select = array.expressions[0]

        # Convert the UNNEST operation in the FROM clause
        unnest = select.args["from"].this
        if isinstance(unnest, sge.Unnest):
            # Convert the column being unnested
            unnest_col = convert(unnest.expressions[0], catalog, compiler)

            # The result of UNNEST is already the elements we need
            result = unnest_col.sort()
    else:
        # For simple arrays without Select statements
        exprs = [convert(expr, catalog, compiler) for expr in array.expressions]
        result = ibis.array(exprs)

    return result


@convert.register(sge.Anonymous)
def convert_anonymous(func, catalog, compiler):
    col = convert(func.expressions[0], catalog, compiler)
    params = [
        convert(expr, catalog=catalog, compiler=compiler)
        for expr in func.expressions[1:]
    ]

    mapper = dict()

    for op_mapper in [SQLGlotCompiler.SIMPLE_OPS, compiler.SIMPLE_OPS]:
        for k, v in op_mapper.items():
            mapper[v] = k

    mapper["ltrim"] = ops.LStrip
    mapper["rtrim"] = ops.RStrip
    mapper["trim"] = ops.Strip

    # TODO: discuss with ibis team a way to store this in the compiler class
    mapper["make_date"] = ops.DateFromYMD
    mapper["make_time"] = ops.TimeFromHMS

    # NOTE: SG passes whitespace params to trim and variants...
    if "trim" in func.this.lower():
        params = []

    op = mapper.get(func.this.lower())

    # if op is None:
    #     res = getattr(col, func.this.lower())(*params)

    # else:
    #     res = op(col, *params).to_expr()
    # return res
    return op(col, *params).to_expr()


@public
def to_sql(
    expr: ir.Expr, dialect: str | None = None, pretty: bool = True, **kwargs
) -> SQLString:
    """Return the formatted SQL string for an expression.

    Parameters
    ----------
    expr
        Ibis expression.
    dialect
        SQL dialect to use for compilation.
    pretty
        Whether to use pretty formatting.
    kwargs
        Scalar parameters

    Returns
    -------
    str
        Formatted SQL string

    """
    import ibis.backends.sql.compilers as sc

    # try to infer from a non-str expression or if not possible fallback to
    # the default pretty dialect for expressions
    if dialect is None:
        try:
            compiler_provider = expr._find_backend(use_default=True)
        except com.IbisError:
            # default to duckdb for SQL compilation because it supports the
            # widest array of ibis features for SQL backends
            compiler_provider = sc.duckdb
    else:
        try:
            compiler_provider = getattr(sc, dialect)
        except AttributeError as e:
            raise ValueError(f"Unknown dialect {dialect}") from e

    if (compiler := getattr(compiler_provider, "compiler", None)) is None:
        raise NotImplementedError(f"{compiler_provider} is not a SQL backend")

    out = compiler.to_sqlglot(expr.unbind(), **kwargs)
    queries = out if isinstance(out, list) else [out]
    dialect = compiler.dialect
    sql = ";\n".join(query.sql(dialect=dialect, pretty=pretty) for query in queries)
    return SQLString(sql)
