from utils.align import run_alignment
from utils.get_chain_info import AntiBody, AntiBodySingle, PairAntiBody
import os
import json

REGION_NAME = ["FRONT", "FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4", "BACK"]


def para_temp_fas(input_dir, ab):
    seq = ab.seq
    regions_list = seq.split("*")
    regions_index_list = []

    # TODO: Handle the situation where certain regions might be missing
    count_length = 0
    for region in regions_list:
        region = region.replace("-", "")
        current_region_length = len(region)
        regions_index_list.append([count_length, count_length + current_region_length])
        count_length += current_region_length

    result_dict = {}
    result_dict["chain_type"] = ab.chain_type
    result_dict[REGION_NAME[0]] = [0, 0]
    result_dict[REGION_NAME[-1]] = [count_length, count_length]
    result_dict["length"] = count_length
    antibody_region = REGION_NAME[1:-1]
    for i in range(len(antibody_region)):
        result_dict[antibody_region[i]] = regions_index_list[i]

    with open(os.path.join(input_dir, "region_index.json"), "w") as f:
        json.dump(result_dict, f, indent=4)


def write_regions(data_dir, antibody_list, args):
    antibody_list = sum([ab.get_all_antibodies() for ab in antibody_list if ab.get_all_antibodies() != None], [])
    
    single_antibody_list = []
    for ab in antibody_list:
        if isinstance(ab, PairAntiBody):
            single_antibody_list.extend(ab.get_all_antibodies())
        else:
            single_antibody_list.append(ab)

    for ab in single_antibody_list:
        temp_dir = os.path.join(data_dir, ab.name)
        if args.use_precomputed_alignments and os.path.exists(
            os.path.join(temp_dir, "region_index.json")
        ):
            continue

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        para_temp_fas(temp_dir, ab)
