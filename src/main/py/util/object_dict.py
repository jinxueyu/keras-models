from typing import Dict, Any


class ObjectDict(Dict[str, Any]):
    """Makes a dictionary behave like an object, with attribute-style access.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # if type(value) is dict:
        #     value = ObjectDict(value)
        #
        # print(name, value)
        # if type(value) is list:
        #     val_list = []
        #     for val in value:
        #         if type(val) is dict:
        #             val = ObjectDict(val)
        #         val_list.append(val)
        #     value = val_list
        self[name] = value


def build_object_dict(dict_value):
    for name, value in dict_value.items():
        if type(value) is dict:
            value = ObjectDict(value)
        if type(value) is list:
            val_list = []
            for val in value:
                if type(val) is dict:
                    val = build_object_dict(val)
                val_list.append(val)
            value = val_list

        dict_value[name] = value

    return ObjectDict(dict_value)
