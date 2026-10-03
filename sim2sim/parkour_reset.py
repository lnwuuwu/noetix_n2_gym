"""Pure state-transition helpers for the MuJoCo parkour runners."""

RESET_SUMMIT = "summit"
RESET_FALL = "fall"
RESET_STUCK = "stuck"
RESET_STAGE_CLEAR = "stage_clear"
RESET_LOW_CLEARANCE = "low_clearance"
RESET_TILT = "tilt"


def validate_course_mode(
        goal_mode, repeat_course, up_only_reset, campaign_mode=False):
    """Reject route modes whose state machines cannot be composed safely."""
    enabled = sum(bool(value) for value in (
        repeat_course, up_only_reset, campaign_mode))
    if enabled > 1:
        raise ValueError(
            "repeat_course, up_only_reset and campaign_mode are "
            "mutually exclusive")
    if up_only_reset and goal_mode != "waypoints":
        raise ValueError("up_only_reset requires goal_mode=waypoints")
    if campaign_mode and goal_mode != "waypoints":
        raise ValueError("campaign_mode requires goal_mode=waypoints")


def update_forward_progress(
        current_x, best_x, last_progress_time, now, min_progress):
    """Update the forward-progress anchor after a meaningful x advance."""
    if min_progress <= 0.0:
        raise ValueError("min_progress must be positive")
    if current_x >= best_x + min_progress:
        return float(current_x), float(now)
    return float(best_x), float(last_progress_time)


def choose_up_only_reset(
        summit_reached, base_height, fall_reset_height,
        seconds_since_progress, no_progress_timeout):
    """Return the reset reason, with successful summits taking precedence."""
    if summit_reached:
        return RESET_SUMMIT
    if base_height < fall_reset_height:
        return RESET_FALL
    if (
            no_progress_timeout > 0.0
            and seconds_since_progress >= no_progress_timeout):
        return RESET_STUCK
    return None


def update_success_hold(condition, hold_started_at, now, hold_seconds):
    """Track how long the final goal condition has remained continuously true."""
    if hold_seconds < 0.0:
        raise ValueError("hold_seconds must be non-negative")
    if not condition:
        return None, False
    started_at = float(now) if hold_started_at is None else float(hold_started_at)
    return started_at, float(now) - started_at >= hold_seconds


def choose_campaign_transition(
        final_goal_stable, relative_base_height, min_base_clearance,
        upright_cos, min_upright_cos, seconds_since_progress,
        no_progress_timeout, in_reset_grace=False):
    """Choose a campaign transition; a stable clear always wins.

    Failure checks are suppressed briefly after a teleport so settling the
    robot onto a new arena cannot consume an attempt.
    """
    if final_goal_stable:
        return RESET_STAGE_CLEAR
    if in_reset_grace:
        return None
    if relative_base_height < min_base_clearance:
        return RESET_LOW_CLEARANCE
    if upright_cos < min_upright_cos:
        return RESET_TILT
    if (
            no_progress_timeout > 0.0
            and seconds_since_progress >= no_progress_timeout):
        return RESET_STUCK
    return None
