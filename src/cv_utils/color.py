class Color:
    def __init__(self, name: str, rgb_value: tuple[int,]) -> None:
        self._validate_name(name)
        self._name = name
        self._validate_rgb(rgb_value)
        self._rgb_value = rgb_value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._validate_name(name)
        self._name = name

    @property
    def rgb_value(self) -> tuple[int,]:
        return self._rgb_value

    @rgb_value.setter
    def rgb_value(self, rgb_value: tuple[int,]) -> None:
        self._validate_rgb(rgb_value)
        self._rgb_value = rgb_value

    def to_dict(self):
        return {self.name: self.rgb_value}

    def __str__(self):
        return f"{self.name:<10}: {self.rgb_value}"

    # --- validators ---

    def _validate_name(self, name: str):
        if not name:
            raise ValueError(f"Invalid name {name!r}")

    def _validate_rgb(self, rgb: tuple):
        if not isinstance(rgb, tuple):
            raise TypeError(f"Invalid type {rgb!r}")
        if not len(rgb) == 3:
            raise ValueError(f"Invalid length of tuple {rgb!r}")


class Palette:
    def __init__(self):
        self._palette = {}
        self._add_roygbiv()

    @property
    def colors(self):
        return self._palette

    def _add_roygbiv(self):
        """Adds rainbow colors to Palette"""

        self.add_color("red", (209, 0, 0))
        self.add_color("orange", (255, 102, 34))
        self.add_color("yellow", (255, 218, 33))
        self.add_color("green", (51, 221, 0))
        self.add_color("blue", (17, 51, 204))
        self.add_color("indigo", (34, 0, 102))
        self.add_color("violett", (51, 0, 68))

    def add_color(self, name: str, rgb_value: tuple):
        self._palette.update(Color(name, rgb_value).to_dict())

    def get_color(self, name: str):
        return self._palette.get(name)

    def __str__(self):
        lines = [f"{name:<10}: {value}" for name, value in self._palette.items()]
        return "\n".join(lines)
