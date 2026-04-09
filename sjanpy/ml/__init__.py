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

from .eval import (
    # Data loading
    load_latent,
    load_split_obs,
    # Subsampling
    subsample_indices,
    # kNN graph
    build_knn_graph,
    knn_to_sparse,
    # UMAP
    fit_umap,
    # Batch integration metrics
    batch_asw,
    celltype_asw,
    graph_connectivity,
    leiden_nmi_ari,
    batch_integration_report,
    # scIB benchmark
    scib_metrics,
    # scGraph (Islander) metrics
    scgraph_score,
)

# Backward compatibility: old names from build_dataset
from .h5ad_io import read_obs as read_obs_h5py
from .h5ad_io import read_var as read_var_h5py
