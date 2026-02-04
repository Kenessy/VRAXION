import unittest


from tools.linear_labels_catalog import ARROW, _split_group_child, _to_ascii, labels_to_rows


class LinearLabelsCatalogTests(unittest.TestCase):
    def test_split_group_child_arrow(self) -> None:
        name = f"EVIDENCE LEVEL {ARROW} E1 PROBE"
        group, child, sep = _split_group_child(name)
        self.assertEqual(group, "EVIDENCE LEVEL")
        self.assertEqual(child, "E1 PROBE")
        self.assertEqual(sep, ARROW)

    def test_split_group_child_no_arrow(self) -> None:
        name = "BATCH SIZE"
        group, child, sep = _split_group_child(name)
        self.assertEqual(group, "")
        self.assertEqual(child, "BATCH SIZE")
        self.assertEqual(sep, "")

    def test_ascii_normalization(self) -> None:
        raw = f"A {ARROW} B"
        asc = _to_ascii(raw)
        self.assertIn("->", asc)
        self.assertNotIn(ARROW, asc)

    def test_deterministic_sort(self) -> None:
        labels = [
            {
                "id": "2",
                "parentId": "G",
                "name": f"EVIDENCE LEVEL {ARROW} E2 CHECK",
                "color": "#000",
                "description": "x",
            },
            {
                "id": "1",
                "parentId": "G",
                "name": f"EVIDENCE LEVEL {ARROW} E1 PROBE",
                "color": "#000",
                "description": "x",
            },
        ]
        rows, warns = labels_to_rows(labels)
        self.assertEqual(warns, [])
        # Sorted by group, then child: E1 before E2.
        self.assertEqual(rows[0]["label_child_raw"], "E1 PROBE")
        self.assertEqual(rows[1]["label_child_raw"], "E2 CHECK")


if __name__ == "__main__":
    unittest.main()

