import { motion } from 'framer-motion';
import Image from 'next/image';
import React from 'react';

interface GaugeChartProps {
  value: number;
}

const GaugeArc = ({ value }: { value: number }) => {
  const clampValue = Math.max(0, Math.min(100, value));

  const totalLength = 852.9671; // 실제 path 길이
  const progress = (clampValue / 100) * totalLength;

  return (
    <svg
      width="320"
      height="270"
      viewBox="0 0 355 298"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="absolute top-[10px] left-[17px] overflow-visible"
    >
      {/* 색상 정의 */}
      <defs>
        <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#FF6A3C" />      {/* 빨강 */}
          <stop offset="50%" stopColor="#FFD600" />     {/* 노랑 */}
          <stop offset="100%" stopColor="#00C853" />    {/* 초록 */}
        </linearGradient>
      </defs>

      {/* 배경 arc */}
      <path
        d="M62.0428 293.348C56.4922 298.906 47.4357 298.94 42.3475 292.956C22.7719 269.933 9.33231 242.222 3.41061 212.412C-3.43826 177.934 0.0768284 142.197 13.5114 109.72C26.9459 77.2427 49.6966 49.484 78.8863 29.9541C108.076 10.4241 142.394 0 177.5 0C212.606 0 246.924 10.4241 276.114 29.9541C305.303 49.484 328.054 77.2427 341.489 109.72C354.923 142.197 358.438 177.934 351.589 212.412C345.668 242.222 332.228 269.933 312.652 292.956C307.564 298.94 298.508 298.906 292.957 293.348"
        stroke="#ECEFF1"
        strokeWidth={15}
        fill="none"
        strokeLinecap="round"
      />

      {/* 전경 arc */}
      <motion.path
        d="M62.0428 293.348C56.4922 298.906 47.4357 298.94 42.3475 292.956C22.7719 269.933 9.33231 242.222 3.41061 212.412C-3.43826 177.934 0.0768284 142.197 13.5114 109.72C26.9459 77.2427 49.6966 49.484 78.8863 29.9541C108.076 10.4241 142.394 0 177.5 0C212.606 0 246.924 10.4241 276.114 29.9541C305.303 49.484 328.054 77.2427 341.489 109.72C354.923 142.197 358.438 177.934 351.589 212.412C345.668 242.222 332.228 269.933 312.652 292.956C307.564 298.94 298.508 298.906 292.957 293.348"
        stroke="url(#gaugeGradient)"
        strokeWidth={15}
        strokeDasharray={totalLength}
        strokeDashoffset={totalLength}
        strokeLinecap="round"
        fill="none"
        animate={{
          strokeDashoffset: totalLength - progress,
        }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
      />
    </svg>
  );
};

const GaugeChart: React.FC<GaugeChartProps> = ({ value }) => {
  const clampValue = Math.max(0, Math.min(100, value));
  const angle = (clampValue / 100) * 265 - 60;

  return (
    <div className="relative w-[355px] h-[307px]">
      <GaugeArc value={value} />

      {/* 틱 마크 */}
      <Image
        src="/ticks.svg"
        alt="ticks"
        width={285}
        height={300}
        className="absolute top-[25px] left-[32px]"
        priority
      />

      {/* 20 텍스트 */}
      <div className="absolute left-[60px] top-[156px] text-[#B0BEC5] text-[16px] font-semibold select-none">
        20
      </div>

      {/* 80 텍스트 */}
      <div className="absolute right-[60px] top-[156px] text-[#B0BEC5] text-[16px] font-semibold select-none">
        80
      </div>

      {/* 바늘 */}
      <motion.img
        src="/needle.svg"
        alt="needle"
        className="absolute left-[40px] top-[115px]"
        style={{
          transformOrigin: '140px 65px',
          transformBox: 'fill-box',
        }}
        animate={{ rotate: angle }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        width={169}
        height={94}
      />

      {/* 점수 */}
      <div className="absolute bottom-[16px] w-full text-center text-[28px] font-bold text-neutral-800">
        {value} <span className="text-neutral-400 text-[18px]">/100</span>
      </div>
    </div>
  );
};

export default GaugeChart;
