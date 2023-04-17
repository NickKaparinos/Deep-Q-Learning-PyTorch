"""
Nikos Kaparinos
2023
"""

from os import makedirs
import cv2


def record_videos(agent, env, log_dir: str, num_episodes: int = 5, prefix: str = ''):
    """ Records video of trained agent"""
    video_frames_list = []
    video_dir = f'{log_dir}videos/'
    makedirs(video_dir, exist_ok=True)
    agent.epsilon = 0.0

    i = 0
    obs = env.reset()[0]
    while i < num_episodes:
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        video_frames_list.append(frame)
        if done:
            obs = env.reset()[0]
            i += 1
            encode_video(video_frames_list, video_dir, i, prefix=prefix)
            video_frames_list.clear()


def encode_video(video_frames_list: list, video_dir: str, video_num: int, fps: float = 60.0, prefix: str = '') -> None:
    """ Encodes video frames list to .mp4 video """
    width, height = video_frames_list[0].shape[0], video_frames_list[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{video_dir}{prefix}video{video_num}.mp4', fourcc, fps, (height, width))

    for frame in video_frames_list:
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.destroyAllWindows()
    video.release()
