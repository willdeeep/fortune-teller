# Book of Thoth — Deck Schema

**Source**: https://thothreadings.com
**Attribution**: Content scraped from thothreadings.com under fair use for
personal and educational purposes. Do not redistribute scraped content.

## Card Count

78 cards:
- 22 Major Arcana
- 56 Minor Arcana (14 each in Wands, Cups, Swords, Disks)

## Structured Fields per Card

Each card page on thothreadings.com provides the following labelled sections,
which are extracted by the parser and stored as separate vector chunks:

| Field key      | Heading on page            | Description |
| -------------- | -------------------------- | ----------- |
| `overall`      | (intro before first heading) | General meaning of the card |
| `drive`        | Drive                      | The card's motivating energy |
| `light`        | Light                      | Positive expression / upright themes |
| `shadow`       | Shadow                     | Challenge / shadow expression |
| `reversed`     | Reversed                   | Meaning when card is inverted |
| `keywords`     | Keywords                   | Short thematic words |
| `advice`       | Advice / Instructions      | Practical guidance from the card |
| `question`     | Question                   | Reflective question the card poses |
| `proposal`     | Proposal                   | What the card proposes or suggests |
| `confirmation` | Confirmation               | What the card confirms |
| `affirmation`  | Affirmation                | Positive affirmation statement |

## Embedding Strategy

- One vector chunk per field per card.
- ~78 cards × ~11 sections ≈ ~858 vectors at 384 dimensions.
- Metadata stored alongside each chunk: `deck_id`, `card_id`, `card_name`,
  `section`, `source_url`.
- Retrieval uses cosine similarity via DuckDB HNSW index.

## Card ID Format

Slugified card name, matching the URL path on thothreadings.com:

| Card name           | ID                   |
| ------------------- | -------------------- |
| The Fool            | `the-fool`           |
| The Magus           | `the-magus`          |
| Ace of Wands        | `ace-of-wands`       |
| Queen of Cups       | `queen-of-cups`      |
| The High Priestess  | `the-high-priestess` |
| Knight of Disks     | `knight-of-disks`    |

## Orientation

Each card can be dealt UPRIGHT or REVERSED (50/50 random during auto-deal).
When reversed, the `reversed` section chunk is weighted in retrieval.

## robots.txt Check

Before running the scraper, verify `https://thothreadings.com/robots.txt`
permits crawling of card and spread paths. Document the result here.

_Status_: TODO — check before first `ft-scrape` run.
