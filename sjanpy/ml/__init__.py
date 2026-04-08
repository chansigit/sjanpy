from .build_dataset import (
    # Gene filtering
    load_gene_list,
    resolve_gene_indices,
    # Condition DSL
    parse_numerical_spec,
    parse_cat_spec,
    apply_transforms,
    build_condition_schema,
    build_condition_tensor,
    # Condition schema I/O
    save_condition_schema,
    load_condition_schema,
    # Core processing
    process_file,
    process_file_safetensors,
    build_dataset,
)

from .h5ad_io import (
    read_obs,
    read_var,
    locate_matrix,
    get_matrix_shape,
    read_matrix_rows,
    read_sparse_chunk,
    validate_matrix_values,
)

from .standardize import (
    build_standardized_h5ads,
    build_standardized_obs,
)

from .dataset import (
    GPUDataset,
    StreamingDataset,
)

# Backward compatibility: old names from build_dataset
from .h5ad_io import read_obs as read_obs_h5py
from .h5ad_io import read_var as read_var_h5py
