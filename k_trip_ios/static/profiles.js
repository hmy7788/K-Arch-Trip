// Korean content from SmartFactory_Vision_Project1/src/explanation/class_profiles.yaml.
// English is a translation of the same world-building profiles.
const architectureProfiles = {
  "straight": {
    "ko": {
      "title": "수직 직선형 건축",
      "rule": "아래 폭이 위 폭의 95% 이상 — 위아래가 같은 수직 외벽",
      "features": [
        "곧은 외벽",
        "균일한 폭",
        "수직 강조",
        "평평한 상단"
      ],
      "design": "지면에서 꼭대기까지 폭이 줄지 않는 수직 외벽. 층마다 같은 평면을 그대로 쌓아 올리기 때문에 구조와 설비가 단순해지고, 엘리베이터·계단을 묶은 코어를 가운데 두면 한 장의 도면을 전 층에 반복할 수 있다. 유리 커튼월 외피와 가장 잘 맞는 형태이기도 하다.",
      "movement": "근대 건축의 \"형태는 기능을 따른다\"와 국제주의 양식(International Style)의 직계. 장식을 걷어내고 구조 자체를 외형으로 드러낸다. 한국에서는 1970~80년대 도심 업무지구가 이 형태로 스카이라인을 만들었고, 지금도 오피스 타워의 기본값이다.",
      "products": "여의도 파크원 타워와 삼성동 트레이드타워가 대표적이다. 전통 건축에서는 경복궁 자경전 십장생 굴뚝처럼 위아래 폭이 같게 쌓아 올린 조적 구조가 같은 계열에 놓인다."
    },
    "en": {
      "title": "Straight Architecture",
      "rule": "The base is at least 95% as wide as the top: vertical walls with an almost uniform width.",
      "features": [
        "Straight walls",
        "Uniform width",
        "Vertical emphasis",
        "Flat top"
      ],
      "design": "Vertical walls keep the same width from ground to roof. Repeating the same floor plan simplifies the structure and building services. A central core containing elevators and stairs allows one plan to repeat throughout the building. This form also works naturally with a glass curtain wall.",
      "movement": "A direct descendant of modern architecture’s “form follows function” and the International Style. Ornament is removed and the structure becomes the exterior expression. In Korea, this form shaped downtown business districts in the 1970s and 1980s and remains a standard for office towers.",
      "products": "Parc1 Tower in Yeouido and Trade Tower in Samseong-dong are representative examples. In traditional architecture, the Ten Symbols of Longevity Chimney at Jagyeongjeon, Gyeongbokgung Palace, belongs to the same family of masonry structures with a consistent width."
    }
  },
  "taper_smooth": {
    "ko": {
      "title": "곡선 테이퍼형 건축",
      "rule": "꺾임점 없이 매끄럽게 좁아짐",
      "features": [
        "끊김 없는 외곽선",
        "위로 갈수록 좁아짐",
        "넓은 하부",
        "상승하는 실루엣"
      ],
      "design": "아래가 넓고 위로 갈수록 끊김 없이 좁아지는 실루엣. 바람을 흘려보내 고층부가 받는 풍하중을 줄이고, 넓은 하부가 그대로 기초 역할을 해 구조적으로 안정적이다. 곡률이 일정해 어느 방향에서 봐도 같은 인상을 준다.",
      "movement": "안정과 상승을 한 형태로 말하기 때문에 오래전부터 기념비와 전망탑이 이 형태를 택했다. 신라의 석조 천문대에서 현대 초고층까지, 같은 구조 원리가 천 년을 건너 반복된다.",
      "products": "경주 첨성대가 가장 이른 사례다 — 돌을 원통형으로 쌓되 위로 갈수록 좁혀 올렸다. 현대에서는 롯데월드타워와 N서울타워 몸통이 같은 실루엣을 그린다."
    },
    "en": {
      "title": "Smooth Taper Architecture",
      "rule": "A smoothly narrowing silhouette without abrupt changes in direction.",
      "features": [
        "Continuous outline",
        "Narrows upward",
        "Broad base",
        "Rising silhouette"
      ],
      "design": "A wide base narrows continuously toward the top. The shape lets wind flow around it, reducing wind loads at greater heights, while the broad lower body provides structural stability. Consistent curvature creates a similar impression from every direction.",
      "movement": "Because this form expresses stability and upward movement together, monuments and observation towers have long adopted it. From Silla’s stone observatory to modern skyscrapers, the same structural principle recurs across a thousand years.",
      "products": "Cheomseongdae Observatory in Gyeongju is an early example, built from circular layers of stone that narrow upward. Lotte World Tower and the shaft of N Seoul Tower offer modern examples of the same silhouette."
    }
  },
  "taper_step": {
    "ko": {
      "title": "계단식 적층형 건축",
      "rule": "뚜렷한 꺾임이 1회 이상",
      "features": [
        "층마다 단차",
        "폭이 단계적 축소",
        "수평 그림자선",
        "기단 위 적층"
      ],
      "design": "넓은 아래층 위에 좁은 층을 얹어 폭이 단계적으로 줄어드는 구조. 층이 바뀌는 자리마다 수평선(지붕·난간·처마)이 생기고 그 아래로 그림자가 진다. 무게를 아래로 모으면서 위층 면적만 줄여 안정성을 얻는 방식이다.",
      "movement": "동아시아 석탑의 기본 문법이다. 기단 위에 탑신을 층층이 올리되 위로 갈수록 폭과 높이를 함께 줄여 상승감을 만든다. 현대에는 일조권 사선제한을 맞추려 층마다 뒤로 물러나는 계단식 후퇴(setback) 형태로 도심에 다시 나타났다.",
      "products": "불국사 다보탑과 석가탑, 익산 미륵사지 석탑, 경천사지 십층석탑이 대표적이다. 미륵사지 석탑은 현존하는 가장 오래되고 가장 큰 석탑으로 꼽힌다. 현대에서는 층마다 물러나는 setback 오피스 빌딩이 같은 계열이다."
    },
    "en": {
      "title": "Step Taper Architecture",
      "rule": "At least one distinct change in the outline creates a visible step.",
      "features": [
        "Tiered setbacks",
        "Width reduces in steps",
        "Horizontal shadow lines",
        "Layers above a base"
      ],
      "design": "Narrower stories sit on wider ones, reducing the width in stages. Each transition creates a horizontal line—a roof, parapet or eave—with a shadow underneath. Concentrating mass below while reducing upper-floor area provides stability.",
      "movement": "A fundamental language of East Asian stone pagodas. Stories rise above a base, with both width and height diminishing toward the top. In modern cities, the form reappears as stepped setbacks that respond to daylight-access restrictions.",
      "products": "Dabotap and Seokgatap at Bulguksa Temple, the Stone Pagoda at Mireuksa Temple Site in Iksan, and the Ten-story Stone Pagoda from Gyeongcheonsa Temple Site are representative examples. The Mireuksa pagoda is considered one of Korea’s oldest and largest surviving stone pagodas. Modern setback office buildings belong to the same family."
    }
  },
  "mug": {
    "ko": {
      "title": "처마 확장형 건축",
      "rule": "높이가 폭의 1.5배 이하 — 낮고 넓으며 옆으로 뻗은 구조물이 있다",
      "features": [
        "낮고 넓은 비례",
        "옆으로 뻗은 처마",
        "지붕이 지배적",
        "깊은 그림자"
      ],
      "design": "높이보다 폭이 넓고, 몸체 밖으로 구조물이 돌출된 형태. 텀블러의 손잡이에 해당하는 것이 건축에서는 처마다 — 기둥 밖으로 뻗어 나와 비와 햇빛을 막는다. 낮고 넓은 비례가 그대로 안정감이 된다.",
      "movement": "한국 목조건축의 기본 문법. 지붕이 건물 전체의 인상을 지배하고, 공포(栱包)를 겹쳐 쌓아 처마를 더 멀리 내민다. 처마 깊이는 여름 햇빛은 막고 겨울 햇빛은 들이도록 기후에 맞춰 조절된 결과다.",
      "products": "경복궁 근정전, 숭례문, 수원화성 팔달문이 대표적이다. 영주 부석사 무량수전은 현존하는 가장 오래된 목조건축 중 하나로, 처마와 기둥의 비례가 교과서로 꼽힌다."
    },
    "en": {
      "title": "Eave Architecture",
      "rule": "Height is no more than 1.5 times the width: a low, broad form with a projecting structure.",
      "features": [
        "Low, broad proportions",
        "Projecting eaves",
        "Dominant roof",
        "Deep shadows"
      ],
      "design": "A broad, low body with a structure projecting beyond it. In this architectural world, a tumbler’s handle corresponds to an eave: it extends beyond the columns to protect against rain and sunlight. The low, broad proportions create a sense of stability.",
      "movement": "A fundamental language of Korean wooden architecture. The roof defines the building’s overall impression, while layered bracket sets, or gongpo, extend the eaves further outward. Their depth responds to the climate, blocking summer sunlight while admitting winter sunlight.",
      "products": "Geunjeongjeon Hall at Gyeongbokgung Palace, Sungnyemun Gate and Paldalmun Gate at Suwon Hwaseong are representative examples. Muryangsujeon Hall at Buseoksa Temple in Yeongju is among Korea’s oldest surviving wooden buildings and a classic reference for the proportions of eaves and columns."
    }
  }
};
