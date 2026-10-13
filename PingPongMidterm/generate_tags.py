"""Make the printable AprilTags (36h11) for T-anne-is.

    .venv/bin/python generate_tags.py      -> tags/tag_<id>.png and tags/sheet.png
"""

import os

import cv2
import numpy as np

from vision import TAG_DICT

TAGS = {0: "START", 1: "Level 1 - Club", 2: "Level 2 - Varsity", 3: "Level 3 - Pro",
        4: "Clay court", 5: "Grass court", 6: "Hard court"}


def tag_card(tag_id, label, size=500, margin=70):
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(TAG_DICT), tag_id, size)
    card = cv2.copyMakeBorder(marker, margin, margin + 60, margin, margin, cv2.BORDER_CONSTANT, value=255)
    card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
    text = f"#{tag_id}  {label}"
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
    cv2.putText(card, text, ((card.shape[1] - w) // 2, card.shape[0] - 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
    return card


def main():
    os.makedirs("tags", exist_ok=True)
    cards = []
    for tid, label in TAGS.items():
        card = tag_card(tid, label)
        cv2.imwrite(f"tags/tag_{tid}.png", card)
        cards.append(card)
    blank = np.full_like(cards[0], 255)
    cards += [blank] * (8 - len(cards))
    sheet = np.vstack([np.hstack(cards[i:i + 2]) for i in range(0, 8, 2)])
    cv2.imwrite("tags/sheet.png", sheet)
    print("Saved tags/tag_0.png ... tag_6.png and tags/sheet.png - print them big!")


if __name__ == "__main__":
    main()
