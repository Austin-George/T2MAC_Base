REGISTRY = {}

from .tmac_p2p_comm_controller import VffacMAC as P2PMAC

REGISTRY['tmac_p2p_comm_mac'] = P2PMAC
