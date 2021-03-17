import numpy as np
import scipy.signal as signal


def split_ts_seq(ts_seq, sep_ts):
    """
    正解が存在する区間で、相対位置移動をスプリット。
    (通常はないが、先頭の正解waypoint, sep_ts[0]以前にts_seqがある場合、ts_seqsはsep_tsより一つ多くなる点に注意。)

    Args:
        ts_seq ([type]): 相対位置移動
        sep_ts ([type]): 正解の場所が記載されている時間

    Returns:
        List[np.ndarray]: スプリット結果
    """

    tss = ts_seq[:, 0].astype(float)  # extract time stamp
    unique_sep_ts = np.unique(sep_ts)
    ts_seqs = []
    start_index = 0
    for i in range(0, unique_sep_ts.shape[0]):
        end_index = np.searchsorted(tss, unique_sep_ts[i], side='right') # 境界値の右側が境界
        if start_index == end_index: # たぶん通常、最初のみend_index=0となり、ここに入る。
            continue
        ts_seqs.append(ts_seq[start_index:end_index, :].copy()) # end_indexと指定することで、境界値は範囲内に含まれる。
        start_index = end_index

    # tail data
    if start_index < ts_seq.shape[0]:
        ts_seqs.append(ts_seq[start_index:, :].copy())

    return ts_seqs


def correct_trajectory(original_xys, end_xy):
    """

    :param original_xys: numpy ndarray, shape(N, 2)
    :param end_xy: numpy ndarray, shape(1, 2)
    :return:
    """
    corrected_xys = np.zeros((0, 2))

    A = original_xys[0, :]
    B = end_xy
    Bp = original_xys[-1, :]

    angle_BAX = np.arctan2(B[1] - A[1], B[0] - A[0])
    angle_BpAX = np.arctan2(Bp[1] - A[1], Bp[0] - A[0])
    angle_BpAB = angle_BpAX - angle_BAX
    AB = np.sqrt(np.sum((B - A) ** 2))
    ABp = np.sqrt(np.sum((Bp - A) ** 2))

    corrected_xys = np.append(corrected_xys, [A], 0)
    for i in np.arange(1, np.size(original_xys, 0)):
        angle_CpAX = np.arctan2(original_xys[i, 1] - A[1], original_xys[i, 0] - A[0])

        angle_CAX = angle_CpAX - angle_BpAB

        ACp = np.sqrt(np.sum((original_xys[i, :] - A) ** 2))

        AC = ACp * AB / ABp

        delta_C = np.array([AC * np.cos(angle_CAX), AC * np.sin(angle_CAX)])

        C = delta_C + A

        corrected_xys = np.append(corrected_xys, [C], 0)

    return corrected_xys


def correct_positions(rel_positions, reference_positions):
    """[summary]

    Args:
        rel_positions ([type]): 相対位置移動
        reference_positions ([type]): 正解位置移動

    Returns:
        [type]: [description]
    """
    
    rel_positions_list = split_ts_seq(rel_positions, reference_positions[:, 0])
    if len(rel_positions_list) != reference_positions.shape[0] - 1: # 末尾は終わりがないので消す意図のようだが。。。。
        # print(f'Rel positions list size: {len(rel_positions_list)}, ref positions size: {reference_positions.shape[0]}')
        del rel_positions_list[-1]
    assert len(rel_positions_list) == reference_positions.shape[0] - 1 # 先頭にも始まりがないものがあればassertになるね。

    corrected_positions = np.zeros((0, 3))
    for i, rel_ps in enumerate(rel_positions_list):
        start_position = reference_positions[i]
        end_position = reference_positions[i + 1]
        
        # init
        abs_ps = np.zeros(rel_ps.shape)

        # copy time stamps
        abs_ps[:, 0] = rel_ps[:, 0]
        
        # 絶対位置計算
        # (相対位置移動は0に対する移動の記録なので累積しないといけない)
        abs_ps[0, 1:3] = rel_ps[0, 1:3] + start_position[1:3]
        for j in range(1, rel_ps.shape[0]): 
            abs_ps[j, 1:3] = abs_ps[j-1, 1:3] + rel_ps[j, 1:3]

        # 開始点を先頭に結合
        abs_ps = np.insert(abs_ps, 0, start_position, axis=0)
        
        corrected_xys = correct_trajectory(abs_ps[:, 1:3], end_position[1:3])
        corrected_ps = np.column_stack((abs_ps[:, 0], corrected_xys))
        if i == 0:
            corrected_positions = np.append(corrected_positions, corrected_ps, axis=0)
        else:
            corrected_positions = np.append(corrected_positions, corrected_ps[1:], axis=0)

    corrected_positions = np.array(corrected_positions)

    return corrected_positions


def init_parameters_filter(sample_freq, warmup_data, cut_off_freq=2):
    """フィルタ初期化

    Args:
        sample_freq ([type]): サンプリング周波数(Hz)
        warmup_data ([type]): ???
        cut_off_freq (int, optional): カットオフ周波数(Hz). Defaults to 2.

    Returns:
        [type]: [description]
    """
    order = 4 # フィルタ次数

    # バターワース型IIRフィルタ
    filter_b, filter_a = signal.butter(order, cut_off_freq / (sample_freq / 2), 'low', False)

    # よくわからないけどフィルタを初期状態にしている？
    # note: フィルタ遅延が怒らないように何かしているのかも？
    zf = signal.lfilter_zi(filter_b, filter_a)
    _, zf        = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)
    _, filter_zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)

    return filter_b, filter_a, filter_zf


def get_rotation_matrix_from_vector(rotation_vector):
    """回転ベクトルから回転行列を計算

    Args:
        rotation_vector ([type]): 回転ベクトル

    Returns:
        [type]: 回転行列
    """
    q1 = rotation_vector[0]
    q2 = rotation_vector[1]
    q3 = rotation_vector[2]

    if rotation_vector.size >= 4:
        q0 = rotation_vector[3]
    else:
        q0 = 1 - q1*q1 - q2*q2 - q3*q3
        if q0 > 0:
            q0 = np.sqrt(q0)
        else:
            q0 = 0

    sq_q1 = 2 * q1 * q1
    sq_q2 = 2 * q2 * q2
    sq_q3 = 2 * q3 * q3
    q1_q2 = 2 * q1 * q2
    q3_q0 = 2 * q3 * q0
    q1_q3 = 2 * q1 * q3
    q2_q0 = 2 * q2 * q0
    q2_q3 = 2 * q2 * q3
    q1_q0 = 2 * q1 * q0

    R = np.zeros((9,))
    if R.size == 9:
        R[0] = 1 - sq_q2 - sq_q3 # 1 - (y*y+z*z)
        R[1] = q1_q2 - q3_q0     # x*y
        R[2] = q1_q3 + q2_q0     # x*z

        R[3] = q1_q2 + q3_q0     # x*y
        R[4] = 1 - sq_q1 - sq_q3 # 1 - (x*x+z*z)
        R[5] = q2_q3 - q1_q0     # y*z

        R[6] = q1_q3 - q2_q0     # x*z
        R[7] = q2_q3 + q1_q0     # y*z
        R[8] = 1 - sq_q1 - sq_q2 # 1 - (x*x+y*y)

        R = np.reshape(R, (3, 3))

    # ここはあり得ない。
    elif R.size == 16:
        R[0] = 1 - sq_q2 - sq_q3
        R[1] = q1_q2 - q3_q0
        R[2] = q1_q3 + q2_q0
        R[3] = 0.0

        R[4] = q1_q2 + q3_q0
        R[5] = 1 - sq_q1 - sq_q3
        R[6] = q2_q3 - q1_q0
        R[7] = 0.0

        R[8] = q1_q3 - q2_q0
        R[9] = q2_q3 + q1_q0
        R[10] = 1 - sq_q1 - sq_q2
        R[11] = 0.0

        R[12] = R[13] = R[14] = 0.0
        R[15] = 1.0

        R = np.reshape(R, (4, 4))

    return R


def get_orientation(R):
    """回転行列から方向変化を計算

    Args:
        R ([type]): 回転行列

    Returns:
        [type]: 方向変化
    """
    flat_R = R.flatten()
    values = np.zeros((3,))
    if np.size(flat_R) == 9:
        values[0] = np.arctan2(flat_R[1], flat_R[4])
        values[1] = np.arcsin(-flat_R[7])
        values[2] = np.arctan2(-flat_R[6], flat_R[8])
    else:
        values[0] = np.arctan2(flat_R[1], flat_R[5])
        values[1] = np.arcsin(-flat_R[9])
        values[2] = np.arctan2(-flat_R[8], flat_R[10])

    return values


def compute_steps(acce_datas):
    """ステップ(1歩)情報を加速度情報から計算

    Args:
        acce_datas (list): 加速度センサ

    Returns:
        tuple: (ステップの発生時間, ステップの発生index, ステップ時の加速度情報)
    """
    step_timestamps = np.array([])
    step_indexs = np.array([], dtype=int)
    step_acce_max_mins = np.zeros((0, 4))
    sample_freq = 50 # サンプリング周期(50Hz)
    window_size = 22 # 平均計算用窓長
    low_acce_mag = 0.6 # 山谷判定の閾値
    step_criterion = 1 # ステップ基準判定バージョン(1,2,3まであるが、今は1)
    interval_threshold = 250

    acce_max = np.zeros((2,))
    acce_min = np.zeros((2,))

    # 加速度の大きさの山谷判定結果
    acce_binarys = np.zeros((window_size,), dtype=int)
    acce_mag_pre = 0
    state_flag = 0

    # フィルタ係数計算 + 初期化
    warmup_data = np.ones((window_size,)) * 9.81 # ???
    filter_b, filter_a, filter_zf = init_parameters_filter(sample_freq, warmup_data)

    # 長さ22の窓データ
    acce_mag_window = np.zeros((window_size, 1))

    # detect steps according to acceleration magnitudes
    for i in np.arange(0, np.size(acce_datas, 0)):
        acce_data = acce_datas[i, :] # t,x,y,zの4次元

        # x,y,z軸の二乗和の平方根(大きさ)
        acce_mag = np.sqrt(np.sum(acce_data[1:] ** 2))

        # LPF適用(1sample)
        acce_mag_filt, filter_zf = signal.lfilter(filter_b, filter_a, [acce_mag], zi=filter_zf)
        acce_mag_filt = acce_mag_filt[0]

        # 窓更新
        acce_mag_window = np.append(acce_mag_window, [acce_mag_filt])
        acce_mag_window = np.delete(acce_mag_window, 0)

        # 窓内統計量計算
        mean_gravity  = np.mean(acce_mag_window)
        acce_std      = np.std (acce_mag_window)
        mag_threshold = np.max([low_acce_mag, 0.4 * acce_std]) # 山谷判定の閾値１(low_acce_mag=0.6か標準偏差の0.4倍のどちらか大きい方)

        # detect valid peak or valley of acceleration magnitudes
        acce_mag_filt_detrend = acce_mag_filt - mean_gravity # 平均に対して現加速度がどれくらい大きいか
        if acce_mag_filt_detrend > np.max([acce_mag_pre, mag_threshold]):
            # peak
            acce_binarys = np.append(acce_binarys, [1])
            acce_binarys = np.delete(acce_binarys, 0)
        elif acce_mag_filt_detrend < np.min([acce_mag_pre, -mag_threshold]):
            # valley
            acce_binarys = np.append(acce_binarys, [-1])
            acce_binarys = np.delete(acce_binarys, 0)
        else:
            # between peak and valley
            acce_binarys = np.append(acce_binarys, [0])
            acce_binarys = np.delete(acce_binarys, 0)

        # 加速度大きさの最大値？更新
        if (acce_binarys[-1] == 0) and (acce_binarys[-2] == 1): # 山の直後
            # 最初の山の直後
            if state_flag == 0:
                acce_max[:] = acce_data[0], acce_mag_filt # 時間と大きさの記録
                state_flag = 1
            
            # 最初以降の山の直後
            elif (state_flag == 1) and ((acce_data[0] - acce_max[0]) <= interval_threshold) and (
                    acce_mag_filt > acce_max[1]):
                acce_max[:] = acce_data[0], acce_mag_filt # 時間と大きさの記録
            
            # ここは通らない(state_flagは{0,1}いずれか)
            elif (state_flag == 2) and ((acce_data[0] - acce_max[0]) > interval_threshold):
                acce_max[:] = acce_data[0], acce_mag_filt
                state_flag = 1

        # choose reasonable step criterion and check if there is a valid step
        # (妥当なステップ基準を選択し、有効なステップがあるかどうかを確認します)
        # save step acceleration data: step_acce_max_mins = [timestamp, max, min, variance]
        step_flag = False

        # 今step_criterion==1のため通らない
        if step_criterion == 2:
            if (acce_binarys[-1] == -1) and ((acce_binarys[-2] == 1) or (acce_binarys[-2] == 0)):
                step_flag = True

        # 今step_criterion==1のため通らない
        elif step_criterion == 3:
            if (acce_binarys[-1] == -1) and (acce_binarys[-2] == 0) and (np.sum(acce_binarys[:-2]) > 1):
                step_flag = True

        # かならずここを通る。
        else:
            # 谷の直後
            if (acce_binarys[-1] == 0) and acce_binarys[-2] == -1:
                if (state_flag == 1) and ((acce_data[0] - acce_min[0]) > interval_threshold):
                    acce_min[:] = acce_data[0], acce_mag_filt # 時間と大きさの記録
                    state_flag = 2
                    step_flag = True

                # ここは通らない(state_flagは{0,1}いずれか)
                elif (state_flag == 2) and ((acce_data[0] - acce_min[0]) <= interval_threshold) and (
                        acce_mag_filt < acce_min[1]):
                    acce_min[:] = acce_data[0], acce_mag_filt # 時間と大きさの記録
        if step_flag:
            step_timestamps = np.append(step_timestamps, acce_data[0])
            step_indexs = np.append(step_indexs, [i])

            # 今のステップの最大最小加速度大きさを保存。偏差の2乗も保存。
            # note: acce_data[0]はstep_timestampsの情報と同一
            step_acce_max_mins = np.append(step_acce_max_mins,
                                           [[acce_data[0], acce_max[1], acce_min[1], acce_std ** 2]], axis=0)

        # ひとつ前のサンプルの平均との差
        acce_mag_pre = acce_mag_filt_detrend

    return step_timestamps, step_indexs, step_acce_max_mins


def compute_stride_length(step_acce_max_mins):
    """ステップ(1歩)情報からその1歩の長さ(ストライド)を計算

    Args:
        step_acce_max_mins ([type]): 1歩の区間での加速度の最大、最小値

    Returns:
        [type]: [description]
    """


    stride_lengths = np.zeros((step_acce_max_mins.shape[0], 2))
    k_real = np.zeros((step_acce_max_mins.shape[0], 2))
    step_timeperiod = np.zeros((step_acce_max_mins.shape[0] - 1, )) # -1はwindow_size依存なので注意
    stride_lengths[:, 0] = step_acce_max_mins[:, 0]
    window_size = 2

    # 移動平均計算用
    step_timeperiod_temp = np.zeros((0, ))

    # calculate every step period - step_timeperiod unit: second
    for i in range(0, step_timeperiod.shape[0]):
        # 1stepの時間(msecをsecにする)
        step_timeperiod_data = (step_acce_max_mins[i + 1, 0] - step_acce_max_mins[i, 0]) / 1000
        step_timeperiod_temp = np.append(step_timeperiod_temp, [step_timeperiod_data])

        # window size = 2 の移動平均
        if step_timeperiod_temp.shape[0] > window_size:
            step_timeperiod_temp = np.delete(step_timeperiod_temp, [0])
        step_timeperiod[i] = np.sum(step_timeperiod_temp) / step_timeperiod_temp.shape[0]

    # 定数値が多いけど物理的意味はあるのかな。。。
    K = 0.4
    K_max = 0.8
    K_min = 0.4
    para_a0 = 0.21468084 # 線形近似係数？
    para_a1 = 0.09154517 # 線形近似係数？
    para_a2 = 0.02301998 # 線形近似係数？

    # calculate parameters by step period and acceleration magnitude variance
    # 加速度の分散と、1stepの時間でk_realという値を推定？
    # - 加速度の分散が大きければ、k_realは大きくなる。(比例)
    # - 1stepの時間が小さければ、k_realは大きくなる。(反比例)
    k_real[:, 0] = step_acce_max_mins[:, 0] # こっちは何にも使ってないような。。。
    k_real[0, 1] = K
    for i in range(0, step_timeperiod.shape[0]):
        k_real[i + 1, 1] = np.max([(para_a0 + para_a1 / step_timeperiod[i] + para_a2 * step_acce_max_mins[i, 3]), K_min])
        k_real[i + 1, 1] = np.min([k_real[i + 1, 1], K_max]) * (K / K_min)

    # calculate every stride length by parameters and max and min data of acceleration magnitude
    # とりあえず加速度のmax-minが大きい場合は、ストライドが大きいと見なされる。
    # またk_realが大きい場合もストライドは大きくなる。k_realは、加速度の分散が大きい or 1stepの時間が短いとき。
    # 1stepの時間が短い == 脚が早く動いている == 距離が大きくなる、ってことか。
    # この単位が、距離になっているか少し不安が残るパラメータである。
    stride_lengths[:, 1] = np.max([(step_acce_max_mins[:, 1] - step_acce_max_mins[:, 2]),
                                   np.ones((step_acce_max_mins.shape[0], ))], axis=0)**(1 / 4) * k_real[:, 1]

    return stride_lengths


def compute_headings(ahrs_datas):
    """回転ベクトルから方向を求める。

    Args:
        ahrs_datas ([type]): 回転ベクトル

    Returns:
        np.ndarray: [時間, 方向]
    """
    headings = np.zeros((np.size(ahrs_datas, 0), 2))
    for i in np.arange(0, np.size(ahrs_datas, 0)):
        ahrs_data = ahrs_datas[i, :]

        # 回転ベクトルを回転行列に変換
        rot_mat = get_rotation_matrix_from_vector(ahrs_data[1:])

        # 回転行列から方向変化を計算
        azimuth, pitch, roll = get_orientation(rot_mat)

        # 0～2piに正規化
        around_z = (-azimuth) % (2 * np.pi)
        headings[i, :] = ahrs_data[0], around_z
    return headings


def compute_step_heading(step_timestamps, headings):
    """1step単位の時間における方向情報を抽出

    Args:
        step_timestamps ([type]): 1step単位の時間
        headings ([type]): 方向情報

    Returns:
        [type]: [description]
    """
    step_headings = np.zeros((len(step_timestamps), 2))
    step_timestamps_index = 0
    for i in range(0, len(headings)):
        if step_timestamps_index < len(step_timestamps):

            # step_timestampsに一致する時間のheadingsをstep_headingsとする。
            if headings[i, 0] == step_timestamps[step_timestamps_index]:
                step_headings[step_timestamps_index, :] = headings[i, :]
                step_timestamps_index += 1
        else:
            break
    assert step_timestamps_index == len(step_timestamps)

    return step_headings


def compute_rel_positions(stride_lengths, step_headings):
    """相対的な移動距離計算

    Args:
        stride_lengths ([type]): ストライド情報
        step_headings ([type]): 方向情報

    Returns:
        np.ndarray: 移動情報[時刻, x移動距離, y移動距離]
    """
    rel_positions = np.zeros((stride_lengths.shape[0], 3))
    for i in range(0, stride_lengths.shape[0]):
        rel_positions[i, 0] = stride_lengths[i, 0] # 時刻
        rel_positions[i, 1] = -stride_lengths[i, 1] * np.sin(step_headings[i, 1]) # 北がheadingの0度なので x = -R*sinθ
        rel_positions[i, 2] = stride_lengths[i, 1] * np.cos(step_headings[i, 1])  # 北がheadingの0度なので y =  R*cosθ

    return rel_positions


def compute_step_positions(acce_datas, ahrs_datas, posi_datas):
    """step positionの計算

    Args:
        acce_datas ([type]): TYPE_ACCELEROMETERの情報
        ahrs_datas ([type]): TYPE_ROTATION_VECTORの情報
        posi_datas ([type]): TYPE_WAYPOINTの情報

    Returns:
        [type]: [description]
    """

    # ステップ(1歩)情報を加速度情報から計算    
    step_timestamps, step_indexs, step_acce_max_mins = compute_steps(acce_datas)

    # 回転ベクトルから方向を計算
    headings = compute_headings(ahrs_datas)

    # ステップ(1歩)情報からその1歩の長さ(ストライド)を計算
    stride_lengths = compute_stride_length(step_acce_max_mins)

    # 1step単位の時間における方向情報を抽出
    step_headings = compute_step_heading(step_timestamps, headings)

    # 相対的な移動距離計算
    rel_positions = compute_rel_positions(stride_lengths, step_headings)


    step_positions = correct_positions(rel_positions, posi_datas)

    return step_positions
