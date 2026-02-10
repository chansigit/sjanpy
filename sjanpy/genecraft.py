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
    
    If mask_hvg_only is True, it requires that sc.pp.highly_variable_genes 
    has already been run on the adata object.
    """
    # Define core patterns
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
    if remove_ig_var:      active_patterns.append(patterns["IG_Var"])
    if remove_hb:          active_patterns.append(patterns["Hemoglobin"])
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
        # Check if highly_variable column exists instead of computing it
        if 'highly_variable' not in adata.var.columns:
            raise ValueError(
                "The 'highly_variable' column is missing from adata.var. "
                "Please run sc.pp.highly_variable_genes(adata) before calling "
                "filter_human_sc_genes with mask_hvg_only=True."
            )
        
        # Strategy: Keep data but ensure these don't drive clustering/PCA
        adata.var.loc[target_genes, 'highly_variable'] = False
        print(f"Filter Action: Masked {len(target_genes)} genes from HVG list.")
    else:
        # Strategy: Physical removal from the object
        adata = adata[:, ~mask].copy()
        print(f"Filter Action: Removed {len(target_genes)} genes from matrix.")

    return adata

def filter_mouse_sc_genes(
    adata, 
    remove_predicted  = True,    # Gm... and clone-based (AC/AL/AP/AJ/AF/CT/FP/GS/KB/LR)
    remove_non_coding = True,   # Linc, Mir, Snor
    remove_antisense  = True,   # -as antisense RNAs
    remove_ig_var     = True,   # Immunoglobulin regions
    remove_hb         = True,   # Hemoglobin genes
    remove_metallothionein = True, # Mt1, Mt2
    remove_histone    = False,
    remove_mt_encoded = False,  # mt-
    remove_ribo       = False,  # Rp[sl]
    mask_hvg_only     = True,
):
    """Filtering for Mouse (Mus musculus) scRNA-seq data."""
    
    patterns = {
        # Mouse predicted genes often start with Gm
        "Predicted": r'^(Ac|Al|Ap|Aj|Af|Ct|Fp|Gs|Kb|Lr)\d+(\.\d+)?$|^Gm\d+|^LOC\d+',
        "Non_Coding": r'^Linc|^Mir|^Snor',
        "Antisense": r'-as\d*$',
        "IG_Var": r'^Igkv|^Iglv|^Ighv|^Igkc$|^Iglc[1-7]$|^Igh[gadm][1-4]?$',
        "Hemoglobin": r'^Hb[abdegemqz]',
        "Metallothionein": r'^Mt[12]',
        "Histone": r'^(Hist1h|Hist2h|Hist3h|Hist4h)',
        "Mito_Encoded": r'^mt-',
        "Ribosomal": r'^Rp[sl]\d+|^Rplp[012]|^Rpsa$'
    }

    active_patterns = []
    # (Logic for selecting patterns remains the same as your Human function)
    args = [remove_predicted, remove_non_coding, remove_antisense, remove_ig_var, 
            remove_hb, remove_metallothionein, remove_histone, remove_mt_encoded, remove_ribo]
    keys = ["Predicted", "Non_Coding", "Antisense", "IG_Var", "Hemoglobin", 
            "Metallothionein", "Histone", "Mito_Encoded", "Ribosomal"]
    
    for state, key in zip(args, keys):
        if state: active_patterns.append(patterns[key])

    if not active_patterns: return adata

    combined_regex = "|".join(active_patterns)
    mask = adata.var_names.str.contains(combined_regex, regex=True)
    target_genes = adata.var_names[mask]

    if mask_hvg_only:
        if 'highly_variable' not in adata.var.columns:
            raise ValueError("Missing 'highly_variable' column in adata.var.")
        adata.var.loc[target_genes, 'highly_variable'] = False
        print(f"Mouse Filter: Masked {len(target_genes)} genes.")
    else:
        adata = adata[:, ~mask].copy()
        print(f"Mouse Filter: Removed {len(target_genes)} genes.")

    return adata

def filter_rat_sc_genes(
    adata, 
    remove_predicted = True,    # LOC... and clone-based
    remove_non_coding = True,
    remove_antisense  = True,
    remove_ig_var     = True,
    remove_hb         = True,
    remove_metallothionein = True,
    remove_histone    = False,
    remove_mt_encoded = False,
    remove_ribo       = False,
    mask_hvg_only     = True,
):
    """Filtering for Rat (Rattus norvegicus) scRNA-seq data."""
    
    patterns = {
        # Rats use LOC prefixes extensively for predicted genes
        "Predicted": r'^(Ac|Al|Ap|Aj|Af|Ct|Fp|Gs|Kb|Lr)\d+(\.\d+)?$|^LOC\d+|^RGD\d+',
        "Non_Coding": r'^Linc|^Mir|^Snor',
        "Antisense": r'-as\d*$',
        "IG_Var": r'^Igkv|^Iglv|^Ighv|^Igkc$|^Iglc[1-7]$|^Igh[gadm][1-4]?$',
        "Hemoglobin": r'^Hb[abdegemqz]',
        "Metallothionein": r'^Mt[12]',
        "Histone": r'^(Hist1h|Hist2h|Hist3h|Hist4h)',
        "Mito_Encoded": r'^Mt-', # Note: Rat MT genes can sometimes be uppercase depending on the GTF
        "Ribosomal": r'^Rp[sl]\d+|^Rplp[012]|^Rpsa$'
    }

    active_patterns = []
    args = [remove_predicted, remove_non_coding, remove_antisense, remove_ig_var, 
            remove_hb, remove_metallothionein, remove_histone, remove_mt_encoded, remove_ribo]
    keys = ["Predicted", "Non_Coding", "Antisense", "IG_Var", "Hemoglobin", 
            "Metallothionein", "Histone", "Mito_Encoded", "Ribosomal"]
    
    for state, key in zip(args, keys):
        if state: active_patterns.append(patterns[key])

    if not active_patterns: return adata

    combined_regex = "|".join(active_patterns)
    # Using case=False for Rat can be safer as some GTFs are less consistent than Mouse
    mask = adata.var_names.str.contains(combined_regex, regex=True, case=True)
    target_genes = adata.var_names[mask]

    if mask_hvg_only:
        if 'highly_variable' not in adata.var.columns:
            raise ValueError("Missing 'highly_variable' column in adata.var.")
        adata.var.loc[target_genes, 'highly_variable'] = False
        print(f"Rat Filter: Masked {len(target_genes)} genes.")
    else:
        adata = adata[:, ~mask].copy()
        print(f"Rat Filter: Removed {len(target_genes)} genes.")

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