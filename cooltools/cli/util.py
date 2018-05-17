import click


def validate_csv(ctx, param, value, default_column):
    if value is None: return
    file_path, _, field_name = value.partition('::')
    if not field_name:
        field_name = default_column
    elif field_name.isdigit():
        field_name = int(field_name)
    return file_path, field_name
