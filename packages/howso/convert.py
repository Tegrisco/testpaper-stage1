# uncompyle6 version 3.5.1
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Sep  7 2019, 18:27:02) 
# [Clang 10.0.1 (clang-1001.0.46.4)]
# Embedded file name: /root/string_recognize/convert.py
# Compiled at: 2019-05-09 22:29:37
# Size of source mod 2**32: 5069 bytes
import collections, torch

class strLabelConverter(object):
    r"""'Convert between str and label.\n\n    NOTE:\n        Insert `blank` to the alphabet for CTC.\n\n    Args:\n        alphabet (str): set of the possible characters.\n        ignore_case (bool, default=True): whether or not to ignore all of the case.\n    '"""

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []
        for item in text:
            item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)

        text = result
        return (
         torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, 'text with length: {} does not match declared length: {}'.format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[(i - 1)] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0:
                        if not (i > 0 and t[(i - 1)] == t[i]):
                            char_list.append(self.alphabet[(t[i] - 1)])

                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), 'texts with length: {} does not match declared length: {}'.format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode((t[index:index + l]),
                  (torch.IntTensor([l])), raw=raw))
                index += l

            return texts


class StrConverter:

    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [self.dict[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, 'text with length: {} does not match declared length: {}'.format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[(i - 1)] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0:
                        if not (i > 0 and t[(i - 1)] == t[i]):
                            char_list.append(self.alphabet[(t[i] - 1)])

                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), 'texts with length: {} does not match declared length: {}'.format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode((t[index:index + l]),
                  (torch.IntTensor([l])), raw=raw))
                index += l

            return texts
# okay decompiling convert.pyc
