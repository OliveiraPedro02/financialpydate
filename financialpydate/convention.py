from enum import Enum


class Convention(str, Enum):
    """
    forward: Take the first valid day later in time.
    preceding: Take the first valid day earlier in time.
    modifiedfollowing: Take the first valid day later in time unless it is across a Month boundary, in which case to
      take the first valid day earlier in time.
    modifiedpreceding: Take the first valid day earlier in time unless it is across a Month boundary, in which case to
      take the first valid day later in time.
    """

    following = 'following'
    preceding = 'preceding'
    modifiedfollowing = 'modifiedfollowing'
    modifiedpreceding = 'modifiedpreceding'
    unadjusted = 'unadjusted'

    @property
    def inverse(self) -> 'Convention':
        match self:
            case Convention.following:
                return Convention.preceding
            case Convention.preceding:
                return Convention.following
            case Convention.modifiedfollowing:
                return Convention.modifiedpreceding
            case Convention.modifiedpreceding:
                return Convention.modifiedfollowing
            case Convention.unadjusted:
                return Convention.unadjusted
            case _:
                NotImplementedError
