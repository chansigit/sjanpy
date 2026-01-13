import scanpy as sc
import re

def filter_human_sc_genes(
    adata, 
    remove_predicted = True,    # LOC... and clone-based (AC/AL/AP/AJ/AF/CT/FP/GS/KB/LR)
    remove_non_coding = True,   # LINC, MIR, SNOR
    remove_antisense  = True,   # -AS antisense RNAs
    remove_ig_var     = True,   # Immunoglobulin variable regions
    remove_hb         = True,   # Hemoglobin genes
    remove_metallothionein = True, # MT1, MT2 (Metallothioneins)
    remove_histone    = False,  # HIST1H, etc. (Optional)
    remove_mt_encoded = False,  # MT- mitochondrial (Usually kept for QC)
    remove_ribo       = False,  # Ribosomal (Usually kept for QC)
    mask_hvg_only     = True,    # If True, keeps data but sets highly_variable=False
):
    """
    Comprehensive filtering of uninformative genes for human scRNA-seq data.
    Uses refined regex patterns for high precision.
    """
    # Define core patterns using your provided exact matches
    patterns = {
        "Predicted": r'^(AC|AL|AP|AJ|AF|CT|FP|GS|KB|LR)\d+(\.\d+)?$|^LOC\d+',
        "Non_Coding": r'^LINC|^MIR|^SNOR',
        "Antisense": r'-AS\d*$',
        "IG_Var": r'^IGKV|^IGLV|^IGHV|^IGKC$|^IGLC[1-7]$|^IGH[GADM][1-4]?$',
        "Hemoglobin": r'^HB[ABDGEMQZ]',
        "Metallothionein": r'^MT[12]',
        "Histone": r'^(HIST1H|HIST2H|HIST3H|HIST4H)',
        "Mito_Encoded": r'^MT-',
        "Ribosomal": r'^RP[SL]\d+|^RPLP[012]|^RPSA$'
    }

    # Select patterns based on arguments
    active_patterns = []
    if remove_predicted: active_patterns.append(patterns["Predicted"])
    if remove_non_coding: active_patterns.append(patterns["Non_Coding"])
    if remove_antisense:  active_patterns.append(patterns["Antisense"])
    if remove_ig_var:     active_patterns.append(patterns["IG_Var"])
    if remove_hb:         active_patterns.append(patterns["Hemoglobin"])
    if remove_metallothionein: active_patterns.append(patterns["Metallothionein"])
    if remove_histone:    active_patterns.append(patterns["Histone"])
    if remove_mt_encoded: active_patterns.append(patterns["Mito_Encoded"])
    if remove_ribo:       active_patterns.append(patterns["Ribosomal"])

    if not active_patterns:
        return adata

    combined_regex = "|".join(active_patterns)
    mask = adata.var_names.str.contains(combined_regex, regex=True)
    target_genes = adata.var_names[mask]

    if mask_hvg_only:
        # Strategy: Keep data but ensure these don't drive clustering
        if 'highly_variable' not in adata.var.columns:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', check_values=False)
        
        adata.var.loc[target_genes, 'highly_variable'] = False
        print(f"Filter Action: Masked {len(target_genes)} genes from HVG list.")
    else:
        # Strategy: Physical removal
        adata = adata[:, ~mask].copy()
        print(f"Filter Action: Removed {len(target_genes)} genes from matrix.")

    return adata

def get_background_gene_dict(adata):
    """
    Returns an exhaustive dictionary of 'background' gene categories for human datasets.
    Based on nomenclature-based patterns and biological artifacts.
    """
    categories = {
        # Technical & Contamination
        "Mito_Encoded": r'^MT-',
        "Ribosomal": r'^RP[SL]\d+|^RPLP[012]|^RPSA$',
        "Hemoglobin": r'^HB[ABDGEMQZ]',
        
        # Stress & Processing Artifacts
        "Metallothionein": r'^MT[12]',
        "HSP": r'^(HSP[A-Z0-9]|DNAJ[A-Z0-9]|HSPA|HSPB|HSPC|HSPH)',
        "IEG": r'^(FOS|FOSB|JUN|JUNB|JUND|ATF3|EGR1|EGR2|EGR3|IER2|IER3|DUSP1|DUSP2|NR4A1|NR4A2|NR4A3|BTG1|BTG2|KLF2|KLF4)$',
        
        # Proliferation & Chromatin
        "Cell_Cycle": r'^(MKI67|TOP2A|HMGB2|TUBA1B|TUBB|CENP|KIF|CCNA|CCNB|CDC|CDK1|UBE2C|BIRC5|PCNA|TYMS|MCM[2-7]|RRM[12]|STMN1)$',
        "Histone": r'^(HIST1H|HIST2H|HIST3H|HIST4H)',
        
        # Genomic Artifacts & Non-coding
        "Genomic_Clone": r'^(AC|AL|AP|AJ|AF|CT|FP|GS|KB|LR)\d+(\.\d+)?$',
        "Predicted_LOC": r'^LOC\d+',
        "Non_Coding": r'^LINC|^MIR|^SNOR|^SNOU|^GAS5$|-AS\d*$',
        "MALAT1": r'^MALAT1$',
        
        # Identity Artifacts
        "Sex_Linked": r'^XIST$|^TSIX$|^RPS4Y1$|^EIF1AY$|^USP9Y$|^DDX3Y$|^UTY$|^NLGN4Y$|^KDM5D$',
        "Immune_Variable": r'^IGKV|^IGLV|^IGHV|^IGKC$|^IGLC[1-7]$|^IGH[GADM][1-4]?$|^TR[ABGV][V]$',
        "HKG_Classic": r'^GAPDH$|^ACTB$|^B2M$|^EEF1A1$|^TPT1$|^LDHA$|^NONO$|^PPIA$|^UBA52$'
    }

    gene_dict = {}
    for cat, pattern in categories.items():
        matched = adata.var_names[adata.var_names.str.contains(pattern, regex=True)].tolist()
        if matched:
            gene_dict[cat] = matched
            
    return gene_dict