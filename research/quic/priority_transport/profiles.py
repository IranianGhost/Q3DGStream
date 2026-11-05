PROFILES = {
    "clean": [],
    "bandwidth": ["rate", "5mbit"],
    "delay": ["delay", "40ms", "rate", "5mbit"],
    "jitter": ["delay", "40ms", "10ms", "distribution", "normal", "rate", "5mbit"],
    "loss": ["delay", "20ms", "loss", "random", "2%", "rate", "5mbit"],
    "burst": [
        "delay",
        "20ms",
        "loss",
        "gemodel",
        "1%",
        "25%",
        "80%",
        "0.1%",
        "rate",
        "5mbit",
    ],
}
