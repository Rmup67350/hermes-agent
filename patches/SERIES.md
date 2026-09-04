# HER-356 — série minimale sur Hermes v2026.8.31

Cette série décrit exclusivement le candidat construit depuis le tag officiel
`v2026.8.31` (`29112bef099274229cadff79cdff7bf7b99c4b77`). Elle ne contient ni
composant métier Jean, ni moteur AI Factory/Linear-native, ni feedback Telegram
HSCE, ni service Impact UGC/Instagram.

## Base macOS

- `17c6e4df5c664f0b3849f84d2cb259aef76fe325` — retire uniquement l’alias
  contributeur dont le nom entre en collision par la casse sur APFS, en
  conservant le mapping canonique amont.

## Série ordonnée

1. `3cb6efd895` — confinement workspace et transport/staging Git bornés.
2. `d720ee80b7` — identité minimale PID + heure de démarrage pour les workers
   Kanban officiels.
3. `09a8337681` — verrous cron fail-closed et confiance MCP default-deny.
4. `b9b7bc0341` — refus d’une mise à jour destructive quand l’ancestralité du
   checkout local n’est pas prouvée.
5. `0849c9b112` — chaînes de fallback providers imbriquées.
6. `f227a76fb7` — imports paresseux des SDK optionnels déjà installés.
7. `8996614969` — sondes réseau externes opportunistes désactivées par défaut.
8. `65d95a8b35` — fallback Gemini natif préservé dans le client de requête.
9. `38236ce91e` — diagnostics de fermeture portables.
10. `44966df17d` — isolation des tests `cmd_update` vis-à-vis des gateways et
    sauvegardes live.
11. `c3c4a54fef` — forkguard minimal hors zone amont, limité aux invariants
    encore conservés dans le core.

## Garanties externes

La personnalisation métier est portée par le dépôt privé
`jean-hermes-layer`. Le contrat G6 a été vérifié contre son candidat local
`fd081fe436b5dbc830ff55ca860b81a16c3024db` : manifeste toujours épinglé au
tag officiel, API plugin requises présentes, 79 tests verts, installateur
`--dry-run` exit 0 et destinations plugins/skills inchangées.

## Exclusions bloquantes

- Aucun dossier `contrib/impact_ugc_channel` ou
  `contrib/instagram_content_contract` dans cette série.
- Aucun skill HSCE ou contrat Instagram embarqué dans le core.
- Aucun hook de feedback Telegram Jean dans l’adapter officiel.
- Aucun ledger, selector, owner registry ou contrôleur Linear-native.
- Aucun catalogue créatif, traceur TTS local, donnée, secret, LaunchAgent ou
  configuration client.

## Gates G6

```text
pytest tests/forkguard/test_fork_guarantees.py
pytest <tous les tests modifiés/ajoutés par la série>
pytest tests/hermes_cli/test_cmd_update.py
jean-hermes-layer: pytest tests
jean-hermes-layer: python -m installer --hermes-home ~/.hermes --dry-run
```

Les preuves finales doivent aussi inclure la suite canonique depuis un venv
propre au worktree, `git diff --check`, lint, scan secret, rollback jetable et
validation du manifeste de cutover. Toute modification du SHA invalide les
revues G7 précédentes.
