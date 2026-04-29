# src/ai_engine/eco_classifier.py
# ECO opening classifier

import re

# ECO code to name mapping with move patterns
# Patterns are listed from MOST SPECIFIC (longest) to LEAST SPECIFIC (shortest)
# The classifier will match the longest pattern that fits
ECO_PATTERNS = [
    # Ruy Lopez variations
    ("C63", ["e4", "e5", "Nf3", "Nc6", "Bb5", "f5"]),
    ("C65", ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6"]),
    ("C68", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"]),
    ("C70", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]),
    ("C78", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Nf3"]),
    ("C80", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Nf3", "Nf6", "O-O"]),
    ("C84", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]),
    ("C88", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Bb3"]),
    ("C92", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6"]),
    ("C60", ["e4", "e5", "Nf3", "Nc6", "Bb5"]),
    # Italian Game
    ("C57", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Nc3", "Nxe4"]),
    ("C55", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"]),
    ("C54", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3"]),
    ("C53", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]),
    ("C50", ["e4", "e5", "Nf3", "Nc6", "Bc4"]),
    # Other C-opening variations
    ("C47", ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6"]),
    ("C46", ["e4", "e5", "Nf3", "Nc6", "Nc3"]),
    ("C45", ["e4", "e5", "Nf3", "Nc6", "d4"]),
    ("C44", ["e4", "e5", "Nf3", "Nc6", "c3"]),
    ("C42", ["e4", "e5", "Nf3", "Nf6"]),
    ("C41", ["e4", "e5", "Nf3", "d6"]),
    ("C40", ["e4", "e5", "Nf3"]),
    ("C33", ["e4", "e5", "f4", "exf4"]),
    ("C30", ["e4", "e5", "f4"]),
    ("C25", ["e4", "e5", "Nc3"]),
    ("C23", ["e4", "e5", "Bc4"]),
    ("C22", ["e4", "e5", "d4", "d5", "Qxd4"]),
    ("C21", ["e4", "e5", "d4"]),
    ("C20", ["e4", "e5"]),
    # French Defense
    ("C18", ["e4", "e6", "d4", "d5", "Nc3", "dxe4"]),
    ("C17", ["e4", "e6", "d4", "d5", "Nc3", "Bb4"]),
    ("C15", ["e4", "e6", "d4", "d5", "Nc3", "Bb4"]),
    ("C11", ["e4", "e6", "d4", "d5", "Nc3", "Nf6"]),
    ("C10", ["e4", "e6", "d4", "d5", "Nc3"]),
    ("C03", ["e4", "e6", "d4", "d5", "Nd2"]),
    ("C02", ["e4", "e6", "e4"]),
    ("C01", ["e4", "e6", "d4", "d5"]),
    ("C00", ["e4", "e6"]),
    # B-opening variations (Sicilian and others)
    ("B96", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Bg5"]),
    ("B95", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]),
    ("B90", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]),
    ("B80", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e6"]),
    ("B70", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"]),
    ("B54", ["e4", "c5", "Nf3", "d6", "d4"]),
    ("B50", ["e4", "c5", "Nf3", "d6"]),
    ("B44", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "e6"]),
    ("B33", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6"]),
    ("B32", ["e4", "c5", "Nf3", "Nc6", "d4"]),
    ("B30", ["e4", "c5", "Nf3", "Nc6"]),
    ("B27", ["e4", "c5", "Nf3"]),
    ("B23", ["e4", "c5", "Nc3"]),
    ("B22", ["e4", "c5", "c3"]),
    ("B21", ["e4", "c5", "d4", "cxd4", "c3"]),
    ("B20", ["e4", "c5"]),
    ("B18", ["e4", "c6", "d4", "d5", "Nc3", "dxe4"]),
    ("B17", ["e4", "c6", "d4", "d5", "Nc3"]),
    ("B15", ["e4", "c6", "Nc3"]),
    ("B13", ["e4", "c6", "d4", "d5"]),
    ("B12", ["e4", "c6", "d4"]),
    ("B10", ["e4", "c6"]),
    ("B07", ["e4", "d6"]),
    ("B06", ["e4", "g6"]),
    ("B01", ["e4", "d5"]),
    ("B00", ["e4"]),
    # D-opening variations
    ("D43", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6"]),
    ("D37", ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Nf3"]),
    ("D35", ["d4", "d5", "c4", "e6", "Nc3", "Nf6"]),
    ("D30", ["d4", "d5", "c4", "e6"]),
    ("D20", ["d4", "d5", "c4", "dxc4"]),
    ("D17", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "a6"]),
    ("D15", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "a6"]),
    ("D10", ["d4", "d5", "c4", "c6"]),
    ("D07", ["d4", "d5", "Nc3", "Nc6"]),
    ("D06", ["d4", "d5"]),
    ("D00", ["d4", "d5"]),
    # E-opening variations
    ("E90", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3"]),
    ("E80", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4"]),
    ("E76", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "f4"]),
    ("E70", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4"]),
    ("E60", ["d4", "Nf6", "c4", "g6"]),
    ("E32", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "Qd3"]),
    ("E20", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"]),
    ("E12", ["d4", "Nf6", "c4", "e6", "Nf3", "b6"]),
    ("E00", ["d4", "Nf6", "c4", "e6"]),
    ("E52", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3"]),
    ("E45", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3"]),
    ("E46", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3"]),
    # A-opening variations
    ("A45", ["d4", "Nf6", "c4"]),
    ("A46", ["d4", "Nf6", "Nf3"]),
    ("A50", ["d4", "Nf6", "c4"]),
    ("A52", ["d4", "Nf6", "c2", "e4"]),
    ("A80", ["d5"]),
    ("A83", ["d5", "f5", "e4"]),
    ("A40", ["c4"]),
    ("A00", ["Nf3"]),  # Reti
]


def _parse_san_from_pgn(pgn_str: str) -> list[str]:
    """Extract SAN moves from PGN string."""
    # Remove result annotations
    pgn_clean = re.sub(r'\s*1-0\s*$', '', pgn_str)
    pgn_clean = re.sub(r'\s*0-1\s*$', '', pgn_clean)
    pgn_clean = re.sub(r'\s*1/2-1/2\s*$', '', pgn_clean)

    tokens = pgn_clean.split()
    moves = []
    for token in tokens:
        # Skip move numbers (1., 2., etc.)
        if re.match(r'^\d+\.?$', token):
            continue
        # Remove anything after parentheses
        token = re.sub(r'\(.*\)', '', token)
        token = token.strip()
        # Keep only valid SAN moves
        if token and (
            re.match(r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?$', token)
            or re.match(r'^O-O(-O)?[+#]?$', token)
        ):
            moves.append(re.sub(r'[+#]', '', token))
    return moves


def classify_eco(pgn: str) -> dict:
    """
    Classify ECO opening from PGN string.
    Matches longest matching pattern from the ECO table.
    """
    moves = _parse_san_from_pgn(pgn)
    if not moves:
        return {"code": "A00", "name": "Uncommon Opening"}

    # Try to match from most specific (longest pattern) to least specific
    for code, pattern in ECO_PATTERNS:
        if moves[:len(pattern)] == pattern:
            return {"code": code, "name": _get_opening_name(code)}

    return {"code": "A00", "name": "Uncommon Opening"}


def _get_opening_name(code: str) -> str:
    """Get opening name from ECO code."""
    names = {
        "A00": "Uncommon Opening", "A40": "English Opening", "A45": "Trompowsky Attack",
        "A46": "English Opening: Queen's Pawn", "A50": "Indian Defense",
        "A52": "Budapest Gambit", "A80": "Dutch Defense", "A83": "Dutch, Staunton Gambit",
        "B00": "King's Pawn Opening", "B01": "Scandinavian Defense",
        "B06": "Modern Defense", "B07": "Pirc Defense", "B10": "Caro-Kann Defense",
        "B12": "Caro-Kann Defense", "B13": "Caro-Kann, Exchange Variation",
        "B15": "Caro-Kann Defense", "B17": "Caro-Kann, Steinitz Variation",
        "B18": "Caro-Kann, Classical Variation", "B20": "Sicilian Defense",
        "B21": "Sicilian, Smith-Morra Gambit", "B22": "Sicilian Defense: Alapin",
        "B23": "Sicilian, Closed", "B27": "Sicilian Defense",
        "B30": "Sicilian Defense", "B32": "Sicilian Defense",
        "B33": "Sicilian Defense: Sveshnikov", "B40": "Sicilian Defense",
        "B44": "Sicilian, Taimanov", "B50": "Sicilian Defense",
        "B54": "Sicilian Defense", "B70": "Sicilian Dragon Variation",
        "B80": "Sicilian, Scheveningen", "B90": "Sicilian, Najdorf",
        "B96": "Sicilian, Najdorf", "C00": "French Defense",
        "C01": "French, Exchange Variation", "C02": "French, Advance Variation",
        "C03": "French, Tarrasch", "C10": "French Defense", "C11": "French Defense",
        "C15": "French, Winawer", "C17": "French, Winawer", "C18": "French, Winawer",
        "C20": "King's Pawn Game", "C21": "Danish Gambit", "C22": "Center Game",
        "C23": "Bishop's Opening", "C25": "Vienna Game", "C30": "King's Gambit",
        "C33": "King's Gambit Accepted", "C40": "King's Knight Opening",
        "C41": "Philidor Defense", "C42": "Petrov Defense", "C44": "Ponziani Opening",
        "C45": "Scotch Game", "C46": "Three Knights Game",
        "C47": "Four Knights Game", "C50": "Italian Game",
        "C53": "Italian Game, Giuoco Piano", "C54": "Italian Game",
        "C55": "Two Knights Defense", "C57": "Two Knights, Traxler Variation",
        "C60": "Ruy Lopez", "C63": "Ruy Lopez, Schliemann Defense",
        "C65": "Ruy Lopez, Berlin Defense", "C68": "Ruy Lopez, Exchange Variation",
        "C70": "Ruy Lopez", "C78": "Ruy Lopez", "C80": "Ruy Lopez, Open",
        "C84": "Ruy Lopez, Closed", "C88": "Ruy Lopez, Closed",
        "C92": "Ruy Lopez, Closed", "D00": "Queen's Pawn Game",
        "D06": "Queen's Gambit", "D07": "Chigorin Defense",
        "D10": "Slav Defense", "D15": "Slav Defense", "D17": "Slav Defense",
        "D20": "Queen's Gambit Accepted", "D30": "Queen's Gambit Declined",
        "D35": "Queen's Gambit Declined", "D37": "Queen's Gambit Declined",
        "D43": "Semi-Slav Defense", "D45": "Semi-Slav Defense",
        "E00": "Catalan Opening", "E12": "Queen's Indian Defense",
        "E20": "Nimzo-Indian Defense", "E32": "Nimzo-Indian, Classical",
        "E46": "Nimzo-Indian Defense", "E52": "Nimzo-Indian Defense",
        "E60": "King's Indian Defense", "E70": "King's Indian Defense",
        "E76": "King's Indian, Four Pawns Attack",
        "E80": "King's Indian, Samisch", "E90": "King's Indian Defense",
    }
    return names.get(code, "Uncommon Opening")
