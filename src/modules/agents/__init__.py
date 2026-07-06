REGISTRY = {}

from .tmac_p2p_comm_rnn_msg_agent import RnnMsgAgent as P2PMsgAgent

REGISTRY['tmac_p2p_comm_rnn_msg'] = P2PMsgAgent
