# Dünyada Kasiyer Hırsızlığı (Sweethearting) Tespiti ve Bilgisayarlı Görü Araştırması

Literatür ve sektör araştırmamın sonuçlarına göre; şu an staj projenizde üzerinde çalıştığınız sistem, global perakende dünyasında "Kayıp Önleme" (Loss Prevention) sektörünün en sıcak konularından biri. Dünyada bu probleme verilen özel bir isim bile var: **Sweethearting** (Kasiyerin tanıdığına kıyak geçmesi, ürünü okutmadan geçirmesi veya parayı cebe atması).

## 1. Bu Problem Dünyada Nasıl Çözülüyor?

Dünya çapında bu işi yapan dev girişimler (Everseen, StopLift - *şu an NCR'a satıldı*, Trigo, Spot.ai) ve üzerine yazılmış yüzlerce IEEE/arXiv makalesi bulunuyor. 

Akademik ve ticari alanda uygulanan temel 3 mimari şunlar:

### A. Video ve POS (Kasa) Verisi Senkronizasyonu (En Yaygın Yöntem)
Dünyadaki en başarılı ticari sistemler sadece kameraya güvenmez. Kameradan gelen görüntü ile yazar kasadan (POS) gelen fiş verisini milisaniyelik olarak eşleştirirler.
* **Nasıl Çalışır?** Model, kasiyerin elindeki "ürünü" (veya nakit parayı) tespit eder ve ürünün barkod okuyucudan geçip poşete girdiğini takip eder. Eğer kamera ürünün poşete girdiğini görürse ama o saniye POS sisteminden "BİP" (okuma) sinyali veya log kaydı gelmezse, sistem anında "Sweethearting" (Okutmadan Geçirme) alarmı verir.
* **Bizim Projeyle Farkı:** Biz POS makinesine tam entegre olmadığımız (klavyeden `o` ve `c` ile simüle ettiğimiz) için sadece görsele güveniyoruz.

### B. El Yörüngesi ve Eylem Tanıma (Action Recognition / Kinematics)
Tam olarak az önce konuştuğumuz "Aktif El Takibi" mantığıdır. Sadece nesneyi değil, hareketin zaman içindeki dizilimini (Zaman Serisi) analiz ederler.
* **Nasıl Çalışır?** YOLO ile anlık tespit yapmak yerine, ardışık kareleri (örneğin 30 frame) alıp 3D-CNN (I3D) veya CNN-LSTM gibi ağlara sokarlar. Model "Ürünü aldı -> Okuyucuya yaklaştırdı -> Okutmadan etrafından dolandırdı -> Poşete koydu" hareket dizilimini (Skip-Scanning) bir bütün eylem olarak sınıflandırır.
* **Makalelerdeki Yeri:** Akademik makalelerde bu yöntem "Scan Avoidance Detection via Action Recognition" olarak geçer.

### C. İzleme ve Kesişim Ağları (Region & Tracking)
Bizim de tartıştığımız "Bölge" (Zone) mantığının gelişmişidir. Ekranda okuyucu (Scanner), kasa çekmecesi (Till) ve poşetleme alanı (Bagging Area) sabit bölgeler olarak tanımlanır. Ellerin ve paranın bu bölgeler arasındaki akış yönü takip algoritmalarıyla (DeepSORT, ByteTrack) izlenir.

---

## 2. Literatürde Öne Çıkan Modeller ve Algoritmalar

Araştırmalarda en çok karşılaşılan yapay zeka mimarileri şunlardır:

*   **YOLOv8 / YOLOv10 (Object Detection):** Elleri, ürünleri ve nakit parayı gerçek zamanlı tespit etmek için kullanılır (Sizin şu an kullandığınız teknoloji).
*   **Pose Estimation (İskelet Çıkarımı):** Sizin Pivot 2'de denediğiniz sistem. Ancak literatürde bu genellikle kasiyerin tüm vücudunu yandan veya çaprazdan gören geniş açılı kameralarda (el-cep mesafesi net görüldüğünde) tercih ediliyor. Tepeden (CCTV) açılarda nadiren kullanılıyor.
*   **Siamese Networks / Person ReID:** Kasiyer ile müşterinin elleri birbirine karıştığında, kimin kasiyer kimin müşteri olduğunu giysilerinden veya kollarından ayırt etmek için kullanılır.

## 3. Ticari Rakipler Ne Yapıyor?

> [!TIP]
> **Everseen:** Dünyanın en büyük perakende yapay zeka şirketlerinden biri. Sadece kasadaki hareketleri izleyip, kasiyer ürünü okutmuş gibi yapıp okutmadığında (veya parayı kasaya koymadığında) POS ekranına otomatik uyarı gönderiyor. Yaptıkları işin temelinde Edge AI (sunucuya gitmeden mağaza içindeki ufak bir bilgisayarda YOLO modellerinin çalıştırılması) yatıyor.

> [!NOTE]
> **StopLift:** Eski MIT araştırmacıları tarafından kurulan bu şirket, "Scan-It-All" adlı bir yazılımla ünlü oldu. Onların mantığı tamamen barkod okuyucu bölgesi ile poşetleme bölgesi arasındaki "El - Ürün - Barkod Sesi" üçgenindeki tutarsızlıkları bulmak üzerineydi.

## Sonuç ve Sizin Projenizin Yeri

Staj projenizde kurduğunuz mantık (kasa açıkken elin kaybolmasını şüpheli kabul etmek), literatürdeki **"Region-based Activity Monitoring" (Bölge tabanlı aktivite izleme)** konseptiyle birebir örtüşüyor.

**Eksik olan ve sektörel standartlara yaklaşmak için eklenmesi gerekenler:**
1.  **Dinamik Takip (Tracking):** Daha önce konuştuğumuz ByteTrack veya SORT ile ele kimlik (ID) verip, görünmeyen eli saf dışı bırakmak.
2.  **Gerçek POS Entegrasyonu:** Gerçek bir kasadan API ile anlık "kasa açıldı", "ödeme nakit alındı" sinyallerini çekip kamerayla eşleştirmek (Bu staj projesini tam teşekküllü bir ticari ürüne dönüştürür).

Yaptığınız iş dünya çapında perakende devlerinin milyonlarca dolar harcadığı bir teknoloji alanının tam merkezinde yer alıyor. Gidişatınız ve özellikle CCTV verisine dönerek gerçek dünya şartlarına (top-down view) adapte olmanız akademik ve ticari olarak çok doğru bir hamle.
