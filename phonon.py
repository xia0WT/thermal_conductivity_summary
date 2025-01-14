
import phonopy
import msgpack
import os
path = "tcd_phonon_db_need_calculate"
#phonon_truncated = {}

for file in os.listdir(path):
    phonon_dict = {}
    phonon_filepath = os.path.join(path, file, "phonon.yaml")
    force_sets_filepath = os.path.join(path, file, "FORCE_SETS")
    
    ph = phonopy.load(phonon_filepath, force_sets_filename=force_sets_filepath, log_level=0)
    ph.auto_projected_dos()
    ph.auto_total_dos()
    phonon_dict["symbols"] = ph.unitcell.symbols
    phonon_dict["frequency_points"] = ph._pdos.frequency_points.tolist()
    phonon_dict["projected_dos"] = ph._pdos.projected_dos.tolist()
    phonon_dict["total_dos"] = ph._total_dos.dos.tolist()
    #phonon_truncated["_".join(file.split("-")[:2])] = phonon_dict
    del ph
    with open("msgpack_data/{}.msgpack".format("_".join(file.split("-")[:2])), "wb") as f:
        f.write(msgpack.packb(phonon_dict))

