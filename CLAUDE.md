# OpenClaw Server Operations

## Server Access
- SSH: `ssh moltbot@138.197.124.246`
- User: moltbot (has sudo)
- Hostname: moltbot-agents

## Key Paths
- OpenClaw binary: `~/.npm-global/bin/openclaw`
- Config: `~/.openclaw/openclaw.json`
- Jinx workspace: `~/.openclaw/workspace/`
- Thyme workspace: `~/.openclaw/workspace-thyme/`
- Sage workspace: `~/.openclaw/workspace-sage/`
- Shared files: `~/.openclaw/shared/`
- Session log: `~/.openclaw/shared/session-log.md`
- Gateway runs in: screen session called "gateway"
- gog binary: `/home/moltbot/.local/bin/gog`

## Agents
- Jinx (WhatsApp) — Z.ai GLM-4.7-Flash — primary assistant / executive assistant
- Thyme (Telegram) — Claude Haiku 4.5 — revenue/social media (TRR, SkyBox)
- Sage (Discord) — Z.ai GLM-4.7-Flash — travel research

## Rules
- ALWAYS back up config before editing: `cp <file> <file>.bak.$(date +%s)`
- Verify JSON after any config edit: `cat ~/.openclaw/openclaw.json | python3 -m json.tool`
- After config changes: restart gateway in screen session
- Check agent list with: `~/.npm-global/bin/openclaw agents list --bindings`
- agents.list[] is an array, not a dictionary
- Follow Config Change Protocol: inspect structure before editing openclaw.json

## Session Wrap-Up (ALWAYS do this at the end of every session)

When the task is complete or the user says they're done, automatically do both of these:

### 1. Append to session log
Append a dated entry to `~/.openclaw/shared/session-log.md` on the server. Format:

```
---
## Session: YYYY-MM-DD HH:MM UTC
### Completed
- [list what was done]

### Changed Files
- [list files created/modified with brief description]

### Still Pending
- [list anything that wasn't finished or needs follow-up]

### Notes
- [any warnings, gotchas, or context for next session]
---
```

### 2. Email the summary
Send the same summary via email:
```bash
echo "<summary text>" | /home/moltbot/.local/bin/gog gmail send --to brent.boyce@gmail.com --subject "OpenClaw Session Log - $(date +%Y-%m-%d)" --account brent.boyce.ai@gmail.com
```

If the gog email fails, still ensure the session-log.md is written. The log file is the source of truth.
