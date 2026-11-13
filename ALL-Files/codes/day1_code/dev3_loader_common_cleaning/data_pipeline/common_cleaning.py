import html
import re
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from config import config


def _refuse(column, values):
    raise ValueError(f"{column}: cannot read {sorted({str(value) for value in values})}")


def remove_duplicate_records(dataframe):
    return dataframe.drop_duplicates(keep="first").reset_index(drop=True)


def refuse_missing_values(dataframe, required_columns):
    for column in required_columns:
        blanks = int(dataframe[column].isna().sum())
        if blanks:
            raise ValueError(f"{column}: {blanks} rows give no value")
    return dataframe


def refuse_repeated_references(dataframe, reference_column):
    values = dataframe[reference_column].dropna()
    repeated = values[values.duplicated(keep=False)].unique()
    if len(repeated):
        raise ValueError(f"{reference_column}: repeated on more than one row {sorted(repeated)}")
    return dataframe


def remove_rows_missing_target(dataframe, target_column):
    return dataframe[dataframe[target_column].notna()].reset_index(drop=True)


def normalize_category_values(dataframe, value_mappings):
    result = dataframe.copy()
    for column, mapping in value_mappings.items():
        if column not in result:
            continue
        lowered = result[column].str.lower()
        unknown = result[column].notna() & ~lowered.isin(list(mapping))
        if unknown.any():
            _refuse(column, result[column][unknown])
        result[column] = lowered.map(mapping)
    return result


def convert_report_references(references):
    digits = references.str.extract(config.REPORT_REFERENCE_PATTERN, expand=False)
    unreadable = references.notna() & digits.isna()
    if unreadable.any():
        _refuse(references.name, references[unreadable])
    return (config.REPORT_REFERENCE_PREFIX + digits.str.zfill(config.REPORT_REFERENCE_DIGITS)).where(digits.notna())


def refuse_unreadable_identifiers(dataframe, identifier_patterns):
    for column, pattern in identifier_patterns.items():
        values = dataframe[column]
        unreadable = values.notna() & ~values.str.fullmatch(pattern).fillna(False).astype(bool)
        if unreadable.any():
            _refuse(column, values[unreadable])
    return dataframe


def convert_whole_numbers(dataframe, number_patterns):
    result = dataframe.copy()
    for column, pattern in number_patterns.items():
        numbers = result[column].str.extract(pattern, expand=False)
        unreadable = result[column].notna() & numbers.isna()
        if unreadable.any():
            _refuse(column, result[column][unreadable])
        result[column] = pd.to_numeric(numbers).astype(config.WHOLE_NUMBER_DTYPE)
    return result


def _read_value(value, reading):
    for notation, pattern in reading["notations"].items():
        match = re.match(pattern, value)
        if match:
            exact = Decimal(match.group(1)) * reading["multipliers"][notation] / reading["divisors"][notation]
            return exact.quantize(reading["step"], rounding=ROUND_HALF_UP)
    return None


def convert_readings(dataframe, readings):
    result = dataframe.copy()
    for column, reading in readings.items():
        values = result[column]
        converted = values.map(lambda value: _read_value(value, reading) if isinstance(value, str) else None)
        unreadable = values.notna() & converted.isna()
        if unreadable.any():
            _refuse(column, values[unreadable])
        numbers = pd.to_numeric(converted.map(lambda value: None if value is None else str(value)))
        result[column] = numbers.astype(reading["dtype"])
    return result


def treat_values_below_minimum_as_missing(dataframe, minimum_values):
    result = dataframe.copy()
    for column, minimum in minimum_values.items():
        result[column] = result[column].mask(result[column] < minimum)
    return result


def treat_values_above_maximum_as_missing(dataframe, maximum_values):
    result = dataframe.copy()
    for column, maximum in maximum_values.items():
        result[column] = result[column].mask(result[column] > maximum)
    return result


def keep_latest_revision_of_each_report(dataframe):
    reference, revision = config.REPORT_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN
    blank = dataframe[reference].isna() | dataframe[revision].isna()
    if blank.any():
        _refuse(revision, dataframe[reference][blank].fillna("no reference"))
    repeated = dataframe.duplicated([reference, revision], keep=False)
    if repeated.any():
        _refuse(revision, dataframe[reference][repeated])
    latest = dataframe.groupby(reference)[revision].transform("max")
    return dataframe[dataframe[revision] == latest].reset_index(drop=True)


def convert_markup_to_plain_text(texts):
    plain = texts.str.replace(config.MARKUP_PATTERN, "", regex=True).map(html.unescape, na_action="ignore")
    plain = plain.str.replace(config.WHITESPACE_RUN_PATTERN, " ", regex=True).str.strip()
    return plain.astype(config.TEXT_DTYPE)


def lower_text_written_in_capitals(texts):
    in_capitals = texts.str.isupper().fillna(False).astype(bool)
    return texts.where(~in_capitals, texts.str.lower())
