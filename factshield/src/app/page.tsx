'use client';

import { useState } from 'react';
import Image from 'next/image';
import Gauge from '@/components/Gauge';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

export default function Home() {

  const [score, setScore] = useState(1); //팩트 점수 기본값
  const [url, setUrl] = useState('');

  const handleSearch = () => {
    alert(`검색 실행: ${url}`);
    const newScore = Math.floor(Math.random() * 101); //실제 검색 로직 대신 예시 점수
    setScore(newScore);
  };

  const handleReferenceClick = (index: number) => {
    alert(`참고자료 ${index + 1} 클릭됨`);
  };

  return (
    <main className="min-h-screen bg-white border-[8px] border-[#CAC4D0] rounded-[18px] flex justify-center py-10">
      <section className="w-full max-w-[950px] bg-white rounded-[28px] flex flex-col items-center px-4 py-6">

        {/* Header */}
        <div className="w-full flex items-center justify-start h-[48px] px-6">
          <div className="w-[48px] h-[48px] flex justify-center items-center">
            <div className="w-[40px] h-[40px] rounded-full flex justify-center items-center">
              <Image
                src="/shield-icon.svg"
                alt="shield"
                width={24}
                height={24}
              />
            </div>
          </div>
          <h1 className="text-[22px] text-[#1D1B20] font-normal ml-4">FactShield</h1>
        </div>

        {/* URL Input */}
        <div className="w-full mt-6 flex justify-center">
          <div className="w-full max-w-[744px] flex gap-2 items-center">
            <Input
              type="text"
              placeholder="팩트체크할 뉴스 URL을 입력하세요"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="rounded-full px-6 py-5 text-[#1D1B20] bg-[#F3EDF7]"
            />
            <Button onClick={handleSearch} className="rounded-full px-4 h-[48px] bg-[#F3EDF7]">
              <img src="/search_icon.svg" alt="검색" className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Gauge Chart */}
        <div className="relative w-[355px] h-[307px] mt-10">
          <Gauge value={score} />
        </div>

        {/* Reference Section */}
        <div className="w-full mt-10">
          <div className="flex items-center h-[48px] px-6">
            <h2 className="text-[22px] text-[#1D1B20]">참고자료</h2>
          </div>
          <div className="flex gap-6 mt-4 px-6 pb-2 overflow-hidden">
            {[1, 2, 3].map((i, index) => (
              <Card
                key={i}
                onClick={() => handleReferenceClick(index)}
                className="w-[132px] hover:shadow-lg transition cursor-pointer"
              >
                <CardContent className="p-2 flex flex-col">
                  <div className="w-full h-[120px] bg-gray-300 rounded-md" />
                  <div className="mt-2 text-[14px] text-[#1D1B20] font-medium break-words text-center">근거자료 {i}
                     </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

      </section>
    </main>
  );
}
