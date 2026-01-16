import time
import logging

from src.network import Node, Link, Network
from src.DNDP import DNDP
from src.OA import OA


if __name__ == "__main__":
    start = time.time()
    dndp = OA(
        work_dir="data/LangFang/",
        node_file="LangFang_nodes.csv",
        link_file="LangFang_net.csv",
        OD_file="morning_peak_OD.npy",
    )
    dndp.run_workflow()
    end = time.time()
    logging.info(f"Total running time: {end - start} seconds")