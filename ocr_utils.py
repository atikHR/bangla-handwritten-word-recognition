
import numpy as np

# Supported Bangla words
BANGLA_WORDS = [
    "জল", "ফল", "বল", "কল", "দল",
    "তল", "চল", "নল", "মল", "গল",
    "মন", "বন", "ধন", "জন", "ঘর",
    "বর", "কর", "সর", "চর", "পথ",
    "মত", "রত", "রস", "দম", "কম",
    "গম", "লব", "যশ", "নর", "পর",
    "মর", "পল"
]


def decode_word(probabilities, labels, dictionary=None):

    if dictionary is None:
        dictionary = BANGLA_WORDS

    probabilities = np.asarray(probabilities)

    char_to_index = {
        char: int(index)
        for index, char in labels.items()
    }

    # Raw CNN prediction
    raw_word = "".join(
        labels[str(int(idx))]
        for idx in np.argmax(probabilities, axis=1)
    )

    num_chars = len(probabilities)

    # Filter supported words
    candidates = [
        word for word in dictionary
        if len(word) == num_chars
        and all(char in char_to_index for char in word)
    ]

    if not candidates:
        return raw_word, raw_word, []

    word_scores = []

    for word in candidates:
        score = sum(
            np.log(
                probabilities[i, char_to_index[char]] + 1e-10
            )
            for i, char in enumerate(word)
        )

        word_scores.append((word, float(score)))

    word_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    final_word = word_scores[0][0]

    return raw_word, final_word, word_scores[:5]
