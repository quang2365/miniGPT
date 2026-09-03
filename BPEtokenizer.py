from collections import Counter
import re


class BPETokenizer:

    def __init__(
        self,
        num_merges=10
    ):
        self.num_merges = num_merges

        # Các merge rule BPE học được.
        self.merges = []

        # Representation dùng trong quá trình train BPE.
        self.vocab = {}

        # Vocabulary thực sự dùng cho model.
        self.token_to_id = {}
        self.id_to_token = {}

        self.unk_token = "<unk>"


    # =====================================================
    # TÌM CÁC CẶP TOKEN LIỀN NHAU
    # =====================================================

    def _get_pairs(
        self,
        vocab
    ):
        pairs = Counter()

        for word, freq in vocab.items():

            symbols = word.split()

            for i in range(
                len(symbols) - 1
            ):
                pair = (
                    symbols[i],
                    symbols[i + 1]
                )

                pairs[pair] += freq

        return pairs


    # =====================================================
    # MERGE MỘT CẶP TOKEN
    # =====================================================

    def _merge_vocab(
    self,
    vocab,
    pair
):
        new_vocab = {}

        old = " ".join(
            pair
        )

        new = "".join(
            pair
        )

        pattern = re.compile(
            r"(?<!\S)"
            + re.escape(old)
            + r"(?!\S)"
        )

        for word, freq in vocab.items():

            new_word = pattern.sub(
                lambda match: new,
                word
            )

            new_vocab[new_word] = (
                new_vocab.get(
                    new_word,
                    0
                )
                + freq
            )

        return new_vocab


    # =====================================================
    # TRAIN BPE
    # =====================================================

    def fit(
        self,
        text_list
    ):
        """
        Học BPE merge rules từ danh sách câu.
        """

        word_counts = Counter()


        # -----------------------------------------------
        # Đếm tần suất từ
        # -----------------------------------------------

        for text in text_list:

            words = (
                text
                .strip()
                .split()
            )

            word_counts.update(
                words
            )


        # -----------------------------------------------
        # Tách mỗi từ thành character
        #
        # "hello"
        #
        # →
        #
        # "h e l l o </w>"
        # -----------------------------------------------

        self.vocab = {
            " ".join(
                list(word)
            ) + " </w>": freq

            for word, freq
            in word_counts.items()
        }


        self.merges = []


        # -----------------------------------------------
        # Học merge rules
        # -----------------------------------------------

        for _ in range(
            self.num_merges
        ):

            pairs = self._get_pairs(
                self.vocab
            )

            if not pairs:
                break


            most_common = (
                pairs
                .most_common(1)[0][0]
            )


            self.vocab = (
                self._merge_vocab(
                    self.vocab,
                    most_common
                )
            )


            self.merges.append(
                most_common
            )


        # -----------------------------------------------
        # Sau khi train xong:
        # xây vocabulary token thực sự.
        # -----------------------------------------------

        self._build_token_vocab()


    # =====================================================
    # TẠO TOKEN VOCABULARY
    # =====================================================

    def _build_token_vocab(self):
        """
        Tạo:
            token_to_id
            id_to_token

        từ BPE vocabulary sau khi merge.
        """

        token_set = set()


        # self.vocab có dạng:
        #
        # "th e </w>"
        # "h ọ c </w>"
        #
        # Ta lấy từng subword token.
        for word in self.vocab.keys():

            symbols = word.split()

            token_set.update(
                symbols
            )


        # Thêm UNK.
        token_list = [
            self.unk_token
        ] + sorted(token_set)


        self.token_to_id = {
            token: idx
            for idx, token
            in enumerate(token_list)
        }


        self.id_to_token = {
            idx: token
            for token, idx
            in self.token_to_id.items()
        }


    # =====================================================
    # ENCODE MỘT WORD → SUBWORD
    # =====================================================

    def encode_word(
        self,
        word
    ):
        """
        Ví dụ:

        'hello'

        ban đầu:
        ['h','e','l','l','o','</w>']

        sau BPE:
        ['he','ll','o</w>']
        """

        symbols = (
            list(word)
            +
            ["</w>"]
        )


        # Áp dụng merge theo đúng thứ tự đã học.
        for pair in self.merges:

            i = 0

            while (
                i
                <
                len(symbols) - 1
            ):

                if (
                    symbols[i] == pair[0]
                    and
                    symbols[i + 1] == pair[1]
                ):

                    symbols[i] = (
                        pair[0]
                        +
                        pair[1]
                    )

                    symbols.pop(
                        i + 1
                    )

                else:
                    i += 1


        return symbols


    # =====================================================
    # TOKENIZE TEXT → SUBWORD STRINGS
    # =====================================================

    def tokenize(
        self,
        text
    ):
        """
        Trả về subword string tokens.
        Dùng để debug / quan sát tokenizer.
        """

        words = (
            text
            .strip()
            .split()
        )


        tokens = []


        for word in words:

            tokens.extend(
                self.encode_word(
                    word
                )
            )


        return tokens


    # =====================================================
    # ENCODE TEXT → TOKEN IDS
    # =====================================================

    def encode(
        self,
        text
    ):
        """
        Raw text
            ↓
        subword tokens
            ↓
        token IDs
        """

        tokens = self.tokenize(
            text
        )


        unk_id = self.token_to_id[
            self.unk_token
        ]


        ids = [
            self.token_to_id.get(
                token,
                unk_id
            )

            for token in tokens
        ]


        return ids


    # =====================================================
    # DECODE TOKEN IDS → TEXT
    # =====================================================

    def decode(
        self,
        ids
    ):
        """
        Token IDs
            ↓
        subword strings
            ↓
        raw text
        """

        tokens = [
            self.id_to_token[
                int(idx)
            ]

            for idx in ids
        ]


        text = "".join(
            tokens
        )


        text = text.replace(
            "</w>",
            " "
        )


        return text.strip()


    # =====================================================
    # VOCAB SIZE
    # =====================================================

    @property
    def vocab_size(self):

        return len(
            self.token_to_id
        )