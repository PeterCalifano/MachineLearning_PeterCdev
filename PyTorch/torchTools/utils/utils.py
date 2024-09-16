# %% Other auxiliary functions - 09-06-2024
def AddZerosPadding(intNum: int, stringLength: str = 4):
    return f"{intNum:0{stringLength}d}"  # Return strings like 00010
