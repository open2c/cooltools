# schemas of datastructures commonly used in cooltools
# including description DataFrame dtypes/columns definitions
diag_expected_dtypes = {
    "region1": "string",
    "region2": "string",
    "dist": "Int64",
    "n_valid": "Int64",
}

block_expected_dtypes = {
    "region1": "string",
    "region2": "string",
    "n_valid": "Int64",
}

# cooler weight names that are potentially divisive
# cooltools supports only multiplicative weight for now
DIVISIVE_WEIGHTS_4DN = ["KR", "VC", "VC_SQRT"]
