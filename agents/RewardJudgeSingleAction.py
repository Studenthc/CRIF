import torch


def human_feedback(before_can_len, before_target_rank, ask_can_len, ask_target_rank, rec_can_len, rec_target_rank):
    ask_target_rank_margin = before_target_rank - ask_target_rank
    ask_target_can_len_margin = before_can_len - ask_can_len
    rec_target_rank_margin = before_target_rank - rec_target_rank
    rec_target_can_len_margin = before_can_len - rec_can_len
    if before_target_rank < 0:
        raise ValueError('VALUE ERROR!')
    if before_target_rank <= 50:
        if before_target_rank <= 9:
            return False
        if ask_target_rank_margin > rec_target_rank_margin:
            return True
        else:
            return False
    else:
        ask_reward = ask_target_rank_margin + 0.5 * ask_target_can_len_margin
        rec_reward = rec_target_rank_margin + 0.5 * rec_target_can_len_margin
        if ask_reward > rec_reward:
            return True
        else:
            return False
