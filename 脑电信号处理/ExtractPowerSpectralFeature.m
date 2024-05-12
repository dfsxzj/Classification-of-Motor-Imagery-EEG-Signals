%% ��ȡ PSD ����
function [power_features] = ExtractPowerSpectralFeature(eeg_data, srate)
    % �� EEG �ź�����ȡ����������
    %   Parameters:
    %       eeg_data:   [channels, frames] �� EEG �ź�����
    %       srate:      int, ������
    %   Returns:
    %       eeg_segments:   [1, n_features] vector
    %% �����������Ƶ�����źŹ���
    [pxx, f] = pwelch(eeg_data, [], [], 128, srate);
    power_delta = bandpower(pxx, f, [0.5, 4], 'psd');
    power_theta = bandpower(pxx, f, [4, 8], 'psd');
    power_alpha = bandpower(pxx, f, [8, 14], 'psd');
    power_beta = bandpower(pxx, f, [14, 30], 'psd');
    power_gamma = bandpower(pxx, f, [30, 50], 'psd');
    % �� pxx ��ͨ��ά���ϵ�ƽ��ֵ
    mean_pxx = mean(pxx, 1);
    power_features = [power_delta; power_theta; power_alpha; power_beta; power_gamma;mean_pxx];

end



