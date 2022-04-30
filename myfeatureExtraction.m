function [features] = myfeatureExtraction(sig)
    mu= mean(sig);
    v= var(sig);
    P= mean(sig.^2);
    r= rms(sig);
    %     H= wentropy(sig,'log energy');
    features= [mu;v;P;r];
end
