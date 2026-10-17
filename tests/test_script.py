"""Text preparation before generation: tidy speaker turns and the automatic frame limit."""
from red_curtain.adapter import frame_budget, tidy_script
from red_curtain.fish_omni.chunking import split_drama3_text


def test_tidy_script_gives_one_clean_line_per_turn():
    text="<|speaker:1|> Wait, are we running on NVIDIA? \n<|speaker:0|>  Nope.\n\n AMD. \n<|speaker:1|> [laughing] Ha."
    assert split_drama3_text(tidy_script(text))==[
        "<|speaker:1|> Wait, are we running on NVIDIA?\n<|speaker:0|> Nope. AMD.\n<|speaker:1|> [laughing] Ha."]


def test_tidy_script_keeps_untagged_text():
    assert tidy_script("  Hello there.\n")=="Hello there."
    assert split_drama3_text(tidy_script("Intro <|speaker:1|> hi"))==["<|speaker:0|> Intro\n<|speaker:1|> hi"]


def test_frame_budget_follows_text_length():
    assert frame_budget("<|speaker:0|> Hi.")==64+4*len("Hi.")
    assert frame_budget("word "*200)==1024
