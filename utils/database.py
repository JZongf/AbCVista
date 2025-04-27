import os

regions = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
regions_fv = ["HFR1", "HCDR1", "HFR2", "HCDR2", "HFR3", "HCDR3", "HFR4", "LFR1", "LCDR1", "LFR2", "LCDR2", "LFR3", "LCDR3", "LFR4"]
schemes = ["chothia", "kabat", "imgt"]

clone_heavy_database_path = {
                        "chothia": "premsa/chothia_heavy_clonotype_seqs.pkl",
                        # "imgt": "premsa/imgt_heavy_clonotype_seqs.pkl",
                        # "kabat": "premsa/kabat_heavy_clonotype_seqs.pkl",
} # 重链克隆型的数据库路径

clone_light_database_path = {
                        "chothia":"premsa/chothia_light_clonotype_seqs.pkl",
                        # "imgt":"premsa/imgt_light_clonotype_seqs.pkl",
                        # "kabat":"premsa/kabat_light_clonotype_seqs.pkl",
} # 轻链克隆型的数据库路径

heavy_length_databases = {
                          # "kabat": "premsa/kabat_heavy_seqs.pkl",
                          # "imgt": "premsa/imgt_heavy_seqs.pkl",
                          "chothia": "premsa/chothia_heavy_seqs.pkl",
                        # "chothia": "premsa/CH_uniref90_hits.pkl"
}

light_length_databases = {
                          # "kabat": "premsa/kabat_light_seqs.pkl",
                          # "imgt": "premsa/imgt_light_seqs.pkl",
                          "chothia": "premsa/chothia_light_seqs.pkl",
                          # "chothia": "premsa/CL_uniref90_hits.pkl"
}

heavy_ab_database_path = {
    # "bfd_uniref":"premsa/CH_bfd_uniclust_hits.fas",
    # "mgnify":"premsa/CH_mgnify_hits.fas",
    "uniref90":"premsa/CH_uniref90_hits.pkl",
}

light_ab_database_path = {
    # "bfd_uniref":"premsa/CL_bfd_uniclust_hits.fas",
    # "mgnify":"premsa/CL_mgnify_hits.fas",
    "uniref90":"premsa/CL_uniref90_hits.pkl",
}

fv_length_database_path = {
    "paired": "premsa/chothia_fv.pkl"
}

unpaired_database_path = {
  "chothia": "premsa/unpaired_database_chothia",
}


def init_databases_path(database_path):
  global clone_heavy_database_path, clone_light_database_path, \
    heavy_length_databases, light_length_databases, \
    heavy_ab_database_path, light_ab_database_path, \
    fv_length_database_path, unpaired_database_path
  
  for database, path in clone_heavy_database_path.items():
    clone_heavy_database_path[database] = os.path.join(database_path, path)
  
  for database, path in clone_light_database_path.items():
    clone_light_database_path[database] = os.path.join(database_path, path)

  for database, path in heavy_length_databases.items():
    heavy_length_databases[database] = os.path.join(database_path, path)
    
  for database, path in light_length_databases.items():
    light_length_databases[database] = os.path.join(database_path, path)
  
  for database, path in heavy_ab_database_path.items():
    heavy_ab_database_path[database] = os.path.join(database_path, path)
  
  for database, path in light_ab_database_path.items():
    light_ab_database_path[database] = os.path.join(database_path, path)
    
  for database, path in fv_length_database_path.items():
    fv_length_database_path[database] = os.path.join(database_path, path)

  for database, path in unpaired_database_path.items():
    unpaired_database_path[database] = os.path.join(database_path, path)