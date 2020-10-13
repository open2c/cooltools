import os.path as op
import csv
import io
import click


class TabularFilePath(click.Path):
    def __init__(
        self, default_column_index, exists=False, resolve_path=False, allow_dash=False
    ):
        """
        Parameters
        ----------
        default_column : str or int
            Name of desired column or 0-based column index.
        exists : bool
        resolve_path : bool

        Returns
        -------
        path to file, column name or index

        """
        self.default_column_index = default_column_index
        super().__init__(
            exists=exists, resolve_path=resolve_path, allow_dash=allow_dash
        )

    def convert(self, value, param, ctx):
        if value is None:
            return
        file_path, _, field = value.partition("::")
        file_path = super().convert(file_path, param, ctx)
        if not field:
            col = self.default_column_index
        elif field.isdigit():
            col = int(field) - 1  # assume one-based from command line
            if col < 0:
                self.fail('Expected one-based column number, received "0".', param, ctx)
        else:
            col = field
        return file_path, col


def sniff_for_header(file_path, sep="\t", comment="#"):
    """
    Warning: reads the entire file into a StringIO buffer!

    """
    with open(file_path, "r") as f:
        buf = io.StringIO(f.read())

    sample_lines = []
    for line in buf:
        if not line.startswith(comment):
            sample_lines.append(line)
            break
    for _ in range(10):
        sample_lines.append(buf.readline())
    buf.seek(0)

    has_header = csv.Sniffer().has_header("\n".join(sample_lines))
    if has_header:
        names = sample_lines[0].strip().split(sep)
    else:
        names = None

    return buf, names


def validate_csv(ctx, param, value, default_column):
    if value is None:
        return
    file_path, _, field_name = value.partition("::")
    if not op.exists(file_path):
        raise click.BadParameter(
            'Path not found: "{}"'.format(file_path), ctx=ctx, param=param
        )
    if not field_name:
        field_name = default_column
    elif field_name.isdigit():
        field_name = int(field_name)
    return file_path, field_name
