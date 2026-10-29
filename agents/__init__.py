from agents.qcfql import QCFQLAgent
from agents.lps import LPSAgent
from agents.qcmfql import QCMFQLAgent
from agents.meanflow import MEANFLOWAgent
from agents.flow import FLOWAgent
from agents.dsrl import DSRLAgent
from agents.cfgrl import CFGRLAgent

agents = dict(
    qcfql=QCFQLAgent,
    lps=LPSAgent,
    qcmfql=QCMFQLAgent,
    flow=FLOWAgent,
    meanflow=MEANFLOWAgent,
    dsrl=DSRLAgent,
    cfgrl=CFGRLAgent,
)
