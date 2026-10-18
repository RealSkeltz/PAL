"""Episodic memory: a timeline of what happened around the user, tied to their goals.

Unlike the fact-based memories, this one is appended to rather than rewritten:
each update turns the new notes into new episodes, and may only extend the
latest episode when the notes simply continue it. Every episode records which
events and turns it was made from, and which goals it relates to.
"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from pal.llm.base import Turn
from pal.memory import goals as goal_updater
from pal.memory import notes
from pal.response.prompt import about_user
from pal.schemas.episode import Episode, EpisodicMemory

LABEL = "episodes"
SHOWN = 5  # latest episodes shown to the model for continuity

TASK = """You keep a timeline of what happens around someone wearing Buddy, a voice assistant with a microphone and a camera. The camera faces the way the user looks, so the user is never in the frame.

You get the latest episodes, the user's current goals with their ids, and new notes, each with an id. Group the new notes into episodes: each a one-sentence account of one thing that happened, such as a conversation, a step in a task, someone arriving or a decision. Keep concrete details worth remembering later: names, numbers, places, decisions.

For each episode, give the ids of the notes it is made from, who was involved besides the user, and the ids of any goals it relates to. If the first new notes simply continue the latest episode, return it revised as extend_last, listing only the new note ids; otherwise leave extend_last empty. Leave out notes that don't matter for later, such as small talk with Buddy that leads nowhere.

Describe the user's world, not their use of Buddy: "camera" notes are simply what is in front of the user."""


class EpisodeUpdate(BaseModel):
    what: str
    who: list[str]
    notes: list[str] = Field(description="ids of the notes this episode is made from, like n3")
    related_goals: list[str] = Field(description="ids of related goals, like task1, or empty")


class Update(BaseModel):
    extend_last: EpisodeUpdate | None = Field(description="the latest episode revised, if the new notes continue it; else null")
    new: list[EpisodeUpdate]


def system_prompt() -> str:
    about = about_user()
    return TASK + (f"\n\nAbout the user: {about} When others say the user's name, they are talking to the user." if about else "")


def build(current: EpisodicMemory, events, turns, memory):
    note_text, note_ids = notes.numbered_notes(events, turns)
    goals = notes.with_ids(goal_updater.sections(memory.goal_memory))
    goal_text = "\n".join(f"[{gid}] {g.text} [{g.status.value}]" for gid, g in goals) or "(none)"
    text = (f"Latest episodes:\n{_timeline(current.episodes[-SHOWN:])}\n\n"
            f"Current goals:\n{goal_text}\n\nNew notes:\n{note_text}")
    return [Turn(role="user", text=text)], (note_ids, {gid: g.id for gid, g in goals})


def apply(previous: EpisodicMemory, update: Update, context) -> EpisodicMemory:
    note_ids, goal_ids = context
    episodes = list(previous.episodes)

    def make(u: EpisodeUpdate, order: int, base: Episode | None = None) -> Episode | None:
        sources = [note_ids[n] for n in u.notes if n in note_ids]
        if not sources and base is None:
            return None  # an episode has to come from something
        times = [at for _, at in sources] + ([base.at, base.until] if base else [])
        return Episode(
            id=base.id if base else Episode.__dataclass_fields__["id"].default_factory(),
            order=order,
            at=min(times),
            until=max(times),
            what=u.what,
            who=u.who,
            source_events=(base.source_events if base else []) + [sid for sid, _ in sources],
            related_goals=[goal_ids[g] for g in u.related_goals if g in goal_ids],
        )

    if update.extend_last is not None and episodes:
        episodes[-1] = make(update.extend_last, episodes[-1].order, base=episodes[-1])
    for u in update.new:
        episode = make(u, (episodes[-1].order + 1) if episodes else 1)
        if episode is not None:
            episodes.append(episode)
    return EpisodicMemory(episodes=episodes, updated_at=datetime.now(timezone.utc))


def summary(current: EpisodicMemory) -> str:
    return _timeline(current.episodes[-3:], indent="    ")


def _timeline(episodes: list[Episode], indent: str = "") -> str:
    if not episodes:
        return f"{indent}(nothing yet)"
    return "\n".join(f"{indent}#{e.order} {e.at.astimezone():%H:%M}-{e.until.astimezone():%H:%M} {e.what}"
                     + (f" (with {', '.join(e.who)})" if e.who else "") for e in episodes)
