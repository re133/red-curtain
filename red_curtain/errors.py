# From Fish Speech (https://github.com/fishaudio/fish-speech, rev 5415669).
# Fish Audio Community License v1.0, see red_curtain/fish_omni/LICENSE and NOTICE.
class InvalidInputError(ValueError):
    """The request cannot be served as given: bad text, references or length.

    The API server answers these with 400; every other failure is a 500.
    """
