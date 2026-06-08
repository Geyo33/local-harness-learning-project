"""Pure helpers for the disconnect / save-on-exit flow (testable without Gradio)."""


def can_save_episode(store_present: bool, msg_count: int) -> bool:
    """True when save_episode() would actually persist an episode.

    Mirrors EpisodicStore/save_episode's guard: an episodic store must exist
    and the conversation (excluding the system message) must have >= 10 messages.
    """
    return store_present and msg_count >= 10
