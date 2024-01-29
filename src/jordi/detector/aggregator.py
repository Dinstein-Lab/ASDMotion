import numpy as np
import pandas as pd


def unify(df):
    _df = pd.DataFrame(columns=df.columns)

    def calc_weighted_mean_score(rows):
        weights = [(row['end_frame'] - row['start_frame']) / (rows[-1]['end_frame'] - rows[0]['start_frame']) for row in rows]
        scores = [row['stereotypical_score'] for row in rows]
        return np.average(scores, weights=weights)

    n = df.shape[0]
    i = 0
    while i < n:
        curr = df.iloc[i]
        if curr['movement'] == 'Stereotypical':
            merge = [curr]
            j = i + 2
            while j < df.shape[0] - 1:
                next_row = df.iloc[j]
                if next_row['start_frame'] < curr['end_frame']:
                    merge.append(next_row)
                    curr = next_row
                    j += 2
                    i += 2
                else:
                    break
            _df.loc[_df.shape[0]] = [merge[0]['video'], merge[0]['video_full_name'], merge[0]['video_path'], merge[0]['start_time'], merge[-1]['end_time'], merge[0]['start_frame'], merge[-1]['end_frame'],
                                     merge[0]['movement'], merge[0]['calc_date'], merge[0]['annotator'], calc_weighted_mean_score(merge)]
        else:
            start_time = _df.iloc[_df.shape[0] - 1]['end_time'] if i > 0 else curr['start_time']
            end_time = df.iloc[i + 1]['start_time'] if i < n - 1 else curr['end_time']
            start_frame = _df.iloc[_df.shape[0] - 1]['end_frame'] if i > 0 else curr['start_frame']
            end_frame = df.iloc[i + 1]['start_frame'] if i < n - 1 else curr['end_frame']
            _df.loc[_df.shape[0]] = [curr['video'], curr['video_full_name'], curr['video_path'], start_time, end_time, start_frame, end_frame,
                                     curr['movement'], curr['calc_date'], curr['annotator'], curr['stereotypical_score']]
        i += 1
    return _df


def aggregate(df, threshold):
    _df = pd.DataFrame(columns=df.columns)
    df['prediction'] = np.where(df['stereotypical_score'] > threshold, 'Stereotypical', 'NoAction')
    i = 0
    while i < df.shape[0]:
        r = df.iloc[i]
        s, t, c, p = r['start_frame'], r['end_frame'], r['stereotypical_score'], r['prediction']
        _s, _t = r['start_time'], r['end_time']
        score = [c]
        j = i + 1
        while j < df.shape[0]:
            rr = df.iloc[j]
            ss, tt, cc, pp = rr['start_frame'], rr['end_frame'], rr['stereotypical_score'], rr['prediction']
            if p == pp:
                t = tt
                _t = rr['end_time']
                score.append(cc)
            else:
                break
            j += 1
        _df.loc[_df.shape[0]] = [r['video'], r['video_full_name'], r['video_path'], _s, _t, s, t, p, pd.Timestamp.now(), NET_NAME, np.mean(score)]
        i = j
    return unify(_df)
