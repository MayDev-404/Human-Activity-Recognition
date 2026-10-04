# Literature review rows

Each member writes their allocated papers in `docs/lit/<member>.md` (`mayank.md` 4 rows, `ishan.md` 5 rows, `addyan.md` 4 rows, 13 in total). The report builder merges them into Table I of the interim report.

Rules:
- Use only the verified papers allocated to you in `docs/BUILD_SPEC.md` section 9.2, or a paper whose DOI or URL a team member has opened and checked. Never add unverified papers or numbers.
- Write every cell in your own words. Cite primary papers for numbers, not survey tables.
- Keep cells short so the table fits the page budget: Method and Dataset under 12 words, Key Result under 20 words, Relevance under 25 words.
- References are an unnumbered list. The builder numbers them globally, [1] to [13] in table order (M1, then M2, then M3); the dataset citation becomes [14].

## Format

```markdown
| Paper (Author, Year) | Method | Dataset | Key Result | Relevance to Project |
|---|---|---|---|---|
| Ronao & Cho, 2016 | ... | ... | ... | ... |

## References
- C. A. Ronao and S.-B. Cho, "...," *Expert Syst. Appl.*, vol. 59, pp. 235-244, 2016, doi: 10.1016/j.eswa.2016.04.032.
```
