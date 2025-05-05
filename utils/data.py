import re
import struct
from typing import List, Any, Union


CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'


def split_camel_case(w: str) -> List[str]:
    """Splits a camel case string into words."""
    return re.findall(r'[A-Z][a-z]*|[a-z]+', re.sub(r'[^a-zA-Z]', '', w))


def pad_with_zeros(data: List[Any], sizes: List[int]) -> List[Any]:
    """
    Recursively pads a high-dimensional list with zeros.

    :param data: The input high-dimensional list to pad.
    :param sizes: A list of target sizes for each dimension.
    :return: A padded high-dimensional list.
    """
    if not all(isinstance(size, int) and size >= 0 for size in sizes):
        raise ValueError()

    if sizes and len(data) > sizes[0]:
        print(data)
        print(sizes)
        raise ValueError()

    if not sizes:
        return data

    target_size = sizes[0]
    padded = []
    for i in range(target_size):
        if i < len(data):
            padded.append(pad_with_zeros(data[i], sizes[1:]))
        else:
            padded.append(fill_with_zeros(sizes[1:]))
    return padded


def fill_with_zeros(sizes: List[int]) -> Union[int, List[Any]]:
    """
    Fills a high-dimensional structure with zeros.

    :param sizes: A list of sizes for each dimension.
    :return: A list filled with zeros according to the specified sizes.
    """
    if not sizes or len(sizes) < 1:
        return 0

    if len(sizes) == 1:
        return [0] * sizes[0]

    return [
        fill_with_zeros(sizes[1:])
        for _ in range(sizes[0])
    ]


def find_sublist_indices(main: List, sub: List):
    sub_len = len(sub)

    for i in (i for i in range(len(main) - sub_len + 1)
              if main[i:i + sub_len] == sub):
        return i, i + sub_len - 1

    return None


def num_to_fixed_len_bin(num: Union[int, float], len: int = 32) -> List[int]:
    if isinstance(num, int):
        binary_rep = bin(num)[2:]
        return list(binary_rep.zfill(len)[-len:])
    elif isinstance(num, float):
        binary_string = ''.join(format(byte, '08b') for byte in struct.pack('!d', num))
        return list(binary_string[-len:])
    else:
        raise TypeError(f"Unsupported type: {type(num)}. Please provide an int or float.")


def bin_to_int(binary_list: List[int]) -> int:
    return int(''.join(map(str, binary_list)), 2)
