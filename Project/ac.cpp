#include<stdio.h>
#include<stdlib.h>
#include<stdarg.h>
#include<string.h>
#include<math.h>
#ifdef __GNUC__
#include<x86intrin.h>
#define	API
#else
#include<intrin.h>
#define	API __declspec(dllexport)
#endif

#define PROF(...)

const int		magic_ac04='A'|'C'<<8|'0'<<16|'4'<<24;
typedef unsigned long long u64;
bool			set_error(const char *file, int line, const char *msg, ...)
{
	printf("AC error\n%s(%d)\n", file, line);
	if(msg)
	{
		va_list args;
		va_start(args, msg);
		vprintf(msg, args);
		va_end(args);
		printf("\n");
	}
	return false;
}
#define			FAIL(REASON, ...)					return set_error(__FILE__, __LINE__, REASON, ##__VA_ARGS__)
int				floor_log2(unsigned long long n)
{
	int logn=0;
	int sh=(n>=1ULL<<32)<<5;logn+=sh, n>>=sh;
		sh=(n>=1<<16)<<4;	logn+=sh, n>>=sh;
		sh=(n>=1<< 8)<<3;	logn+=sh, n>>=sh;
		sh=(n>=1<< 4)<<2;	logn+=sh, n>>=sh;
		sh=(n>=1<< 2)<<1;	logn+=sh, n>>=sh;
		sh= n>=1<< 1;		logn+=sh;
	return logn;
}
int				ceil_log2(unsigned long long n)
{
	int l2=floor_log2(n);
	l2+=(1ULL<<l2)<n;
	return l2;
}
inline int		clamp(int lo, int x, int hi)
{
	if(x<lo)
		x=lo;
	if(x>hi)
		x=hi;
	return x;
}
inline bool		emit_byte(unsigned char *&out_data, unsigned long long &out_size, unsigned long long &out_cap, unsigned char b)
{
	if(out_size>=out_cap)
	{
		auto newcap=out_cap?out_cap<<1:1;
		auto ptr=(unsigned char*)realloc(out_data, newcap);
		if(!ptr)
			FAIL("Realloc returned nullptr");
		out_data=ptr, out_cap=newcap;
	}
	out_data[out_size]=b;
	++out_size;
	return true;
}
inline bool		emit_pad(unsigned char *&out_data, const unsigned long long &out_size, unsigned long long &out_cap, int size)
{
	while(out_size+size>=out_cap)
	{
		auto newcap=out_cap?out_cap<<1:1;
		auto ptr=(unsigned char*)realloc(out_data, newcap);
		if(!ptr)
			FAIL("Realloc returned nullptr");
		out_data=ptr, out_cap=newcap;
	}
	memset(out_data+out_size, 0, size);
	return true;
}
inline void		store_int_le(unsigned char *base, unsigned long long &offset, int i)
{
	auto p=(unsigned char*)&i;
	base[offset  ]=p[0];
	base[offset+1]=p[1];
	base[offset+2]=p[2];
	base[offset+3]=p[3];
	offset+=4;
}
inline int		load_int_le(const unsigned char *buffer)
{
	int i=0;
	auto p=(unsigned char*)&i;
	p[0]=buffer[0];
	p[1]=buffer[1];
	p[2]=buffer[2];
	p[3]=buffer[3];
	return i;
}
inline int		load_int_be(const unsigned char *buffer)
{
	int i=0;
	auto p=(unsigned char*)&i;
	p[0]=buffer[3];
	p[1]=buffer[2];
	p[2]=buffer[1];
	p[3]=buffer[0];
	return i;
}
	#define		LOG_WINDOW_SIZE		16	//[2, 16]	do not change
	#define		LOG_CONFBOOST		14
	#define		ABAC2_CONF_MSB_RELATION

const double	boost_power=4, min_conf=0.55;
const int		window_size=1<<LOG_WINDOW_SIZE, prob_mask=window_size-1, prob_max=window_size-2, prob_init=(1<<(LOG_WINDOW_SIZE-1))-1;
int				abac4_encode(const void *src, int imsize, int depth, int bytestride, unsigned char *&out_data, unsigned long long &out_size, unsigned long long &out_cap, int loud)
{
	if(!src||!imsize||!depth||!bytestride)
		FAIL("abac4_encode(src=%p, imsize=%d, depth=%d, stride=%d)", src, imsize, depth, bytestride);
	auto buffer=(const unsigned char*)src;
	auto t1=__rdtsc();

	u64 out_start=out_size, out_idx_sizes=out_start+sizeof(int), out_idx_conf=out_idx_sizes+depth*sizeof(int);
	int headercount=1+(depth<<1);
	if(!emit_pad(out_data, out_size, out_cap, headercount*sizeof(int)))
		return false;
	store_int_le(out_data, out_size, magic_ac04);
	out_size=out_idx_conf+depth*sizeof(int);
	
#ifdef AC_MEASURE_PREDICTION
	u64 hitnum=0, hitden=0;//prediction efficiency
#endif

	//std::vector<std::string> planes(depth);
	for(int kp=depth-1;kp>=0;--kp)//bit-plane loop		encode MSB first
	{
		u64 out_planestart=out_size;
		int bit_offset=kp>>3, bit_shift=kp&7;
		int bit_offset2=(kp+1)>>3, bit_shift2=(kp+1)&7;
		//auto &plane=planes[depth-1-kp];
		int prob=0x8000, prob_correct=0x8000;//cheap weighted average predictor

		u64 hitcount=1;

		for(int kb=0, kb2=0;kb<imsize;++kb, kb2+=bytestride)//analyze bitplane
		{
			int bit=buffer[kb2+bit_offset]>>bit_shift&1;
			int p0=((long long)(prob-0x8000)*prob_correct>>16);
			p0+=0x8000;
			//int p0=0x8000+(long long)(prob-0x8000)*hitcount/(kb+1);
			p0=clamp(1, p0, prob_max);
			int correct=bit^(p0>=0x8000);
			//if(kp==1)
			//	printf("%d", bit);//actual bits
			//	printf("%d", p0<0x8000);//predicted bits
			//	printf("%d", !correct);//prediction error
			hitcount+=correct;
			prob=!bit<<15|prob>>1;
			prob_correct=correct<<15|prob_correct>>1;
		}
		PROF(ENC_ANALYZE_PLANE);
		u64 offset=out_idx_conf+kp*sizeof(int);
		store_int_le(out_data, offset, (int)hitcount);
		//out_data[out_idx_conf+kp]=(int)hitcount;//X assigns single byte
		//out_data[out_idx_conf+depth-1-kp]=(int)hitcount;

		if(hitcount<imsize*min_conf)//incompressible, bypass
		{
			emit_pad(out_data, out_size, out_cap, (imsize+7)>>3);
			auto plane=out_data+out_size;
			//plane.resize((imsize+7)>>3, 0);
			for(int kb=0, kb2=0, b=0;kb<imsize;++kb, kb2+=bytestride)
			{
				int byte_idx=kb>>3, bit_idx=kb&7;
				int bit=buffer[kb2+bit_offset]>>bit_shift&1;
			//	int bit=buffer[kb]>>kp&1;
				plane[byte_idx]|=bit<<bit_idx;
			}
			PROF(ENC_BYPASS_PLANE);
		}
		else
		{
			int hitratio_sure=int(0x10000*pow((double)hitcount/imsize, 1/boost_power)), hitratio_notsure=int(0x10000*pow((double)hitcount/imsize, boost_power));
			int hitratio_delta=hitratio_sure-hitratio_notsure;
		//	int hitratio_sure=int(0x10000*cbrt((double)hitcount/imsize)), hitratio_notsure=int(0x10000*(double)hitcount*hitcount*hitcount/((double)imsize*imsize*imsize));
		//	int hitratio_sure=int(0x10000*sqrt((double)hitcount/imsize)), hitratio_notsure=int(0x10000*(double)hitcount*hitcount/((double)imsize*imsize));
			hitcount=(hitcount<<16)/imsize;

			//hitcount=unsigned(((u64)hitcount<<16)/imsize);
			//hitcount=abac2_normalize16(hitcount, logimsize);
			//hitcount*=invimsize;

			prob_correct=prob=0x8000;

#ifdef ABAC2_CONF_MSB_RELATION
			int prevbit0=0;
#endif
			
			emit_pad(out_data, out_size, out_cap, imsize>>8);
			//plane.reserve(imsize>>8);
			unsigned start=0;
			u64 range=0xFFFFFFFF;
			for(int kb=0, kb2=0;kb<imsize;kb2+=bytestride)//bit-pixel loop		http://mattmahoney.net/dc/dce.html#Section_32
			{
				int bit=buffer[kb2+bit_offset]>>bit_shift&1;
			//	int bit=buffer[kb]>>kp&1;
#ifdef ABAC2_CONF_MSB_RELATION
				int prevbit=buffer[kb2+bit_offset2]>>bit_shift2&1;
			//	int prevbit=buffer[kb]>>(kp+1)&1;
#endif
				
				if(range<3)
				{
					//emit_pad(out_data, out_size, out_cap, 4);
					//memcpy(out_data+out_size, &start, 4), out_size+=4;

					emit_byte(out_data, out_size, out_cap, start>>24);//big endian
					emit_byte(out_data, out_size, out_cap, start>>16&0xFF);
					emit_byte(out_data, out_size, out_cap, start>>8&0xFF);
					emit_byte(out_data, out_size, out_cap, start&0xFF);

					//plane.push_back(start>>24);
					//plane.push_back(start>>16&0xFF);
					//plane.push_back(start>>8&0xFF);
					//plane.push_back(start&0xFF);
					start=0, range=0xFFFFFFFF;//because 1=0.9999...
				}
				
				int p0=prob-0x8000;
				p0=p0*prob_correct>>16;
				p0=p0*prob_correct>>16;
				int sure=-(prevbit==prevbit0);
				p0=p0*(hitratio_notsure+(hitratio_delta&sure))>>16;
				//p0=p0*(prevbit==prevbit0?hitratio_sure:hitratio_notsure)>>16;
				//p0=(long long)p0*hitcount>>16;
				p0+=0x8000;
				//if(prevbit!=prevbit0)
				//	p0=0x8000;
				//	p0=0xFFFF-p0;

				//int p0=0x8000+((long long)(prob-0x8000)*(prevbit==prevbit0?hitratio_sure:hitratio_notsure)>>16);

				//int p0=(long long)(prob-0x8000)*sqrthitcount>>16;
				//if(prevbit==prevbit0)
				//	p0=(long long)p0*hitcount>>16;
				//p0+=0x8000;

				//int confboost=prevbit==prevbit0;
				//confboost-=!confboost;
				//confboost<<=LOG_CONFBOOST;
				//int p0=0x8000+((long long)(prob-0x8000)*(hitcount+confboost)>>16);

			//	int p0=0x8000+(int)((prob-0x8000)*(prevbit==prevbit0?sqrt((double)test_conf[kp]/imsize):(double)test_conf[kp]*test_conf[kp]/((double)imsize*imsize)));
			//	int p0=prevbit==prevbit0?prob:0x8000;
			//	int p0=0x8000+(long long)(prob-0x8000)*test_conf[kp]/imsize;
			//	int p0=0x8000+(long long)(prob-0x8000)*hitcount/(kb+1);
				p0=clamp(1, p0, prob_max);
				unsigned r2=(unsigned)(range*p0>>16);
				r2+=(r2==0)-(r2==range);
#ifdef DEBUG_ABAC2
				if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
					printf("%6d %6d %d %08X+%08X %08X %08X\n", kp, kb, bit, start, (int)range, r2, start+r2);
#endif

				int correct=bit^(p0>=0x8000);
			//	hitcount+=correct;
				prob=!bit<<15|prob>>1;
				prob_correct=correct<<15|prob_correct>>1;
#ifdef ABAC2_CONF_MSB_RELATION
				prevbit0=prevbit;
#endif
#ifdef AC_MEASURE_PREDICTION
				hitnum+=correct, ++hitden;
#endif
				auto start0=start;
				if(bit)
				{
					++r2;
					start+=r2, range-=r2;
				}
				//	start=middle+1;
				else
					range=r2-1;
				//	end=middle-1;
				if(start<start0)//
				{
					FAIL("AC OVERFLOW: start = %08X -> %08X, r2 = %08X", start0, start, r2);
					//printf("OVERFLOW\nstart = %08X -> %08X, r2 = %08X", start0, start, r2);
					//int k=0;
					//scanf_s("%d", &k);
				}
				++kb;
				
				while((start^(start+(unsigned)range))<0x1000000)//most significant byte has stabilized			zpaq 1.10
				{
#ifdef DEBUG_ABAC2
					if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
						printf("range 0x%08X byte-out 0x%02X\n", (int)range, start>>24);
#endif
					emit_byte(out_data, out_size, out_cap, start>>24);
					//plane.push_back(start>>24);
					start<<=8;
					range=range<<8|0xFF;
				}
			}
			emit_byte(out_data, out_size, out_cap, start>>24);//big endian
			emit_byte(out_data, out_size, out_cap, start>>16&0xFF);
			emit_byte(out_data, out_size, out_cap, start>>8&0xFF);
			emit_byte(out_data, out_size, out_cap, start&0xFF);

			//plane.push_back(start>>24&0xFF);//big-endian
			//plane.push_back(start>>16&0xFF);
			//plane.push_back(start>>8&0xFF);
			//plane.push_back(start&0xFF);
			PROF(ENC_AC);
		}
		if(loud)
		{
			int c=load_int_le(out_data+out_idx_conf+kp*sizeof(int));
			printf("bit %d: conf = %6d / %6d = %lf%%\n", kp, c, imsize, 100.*c/imsize);
		}
		//	printf("bit %d: conf = %6d / %6d = %lf%%\n", kp, hitcount, imsize, 100.*hitcount/imsize);
		offset=out_idx_sizes+kp*sizeof(int);
		store_int_le(out_data, offset, (int)(out_size-out_planestart));
		//out_data[out_idx_sizes+kp]=out_size-out_planestart;
		//out_data[out_idx_sizes+depth-1-kp]=out_size-out_planestart;
	}
	auto t2=__rdtsc();
	//out_data.clear();
	//for(int k=0;k<depth;++k)
	//	out_sizes[k]=(int)planes[k].size();
	//for(int k=0;k<depth;++k)
	//{
	//	auto &plane=planes[k];
	//	out_data.insert(out_data.end(), plane.begin(), plane.end());
	//}
	//auto t3=__rdtsc();

	if(loud)
	{
		int original_bitsize=imsize*depth, compressed_bitsize=(int)(out_size-out_start)<<3;
		printf("AC encode:  %lld cycles, %lf c/byte\n", t2-t1, (double)(t2-t1)/(original_bitsize>>3));
		printf("Size: %d -> %d, ratio: %lf, %lf bpp\n", original_bitsize>>3, compressed_bitsize>>3, (double)original_bitsize/compressed_bitsize, (double)compressed_bitsize/imsize);
#ifdef AC_MEASURE_PREDICTION
		printf("Predicted: %6lld / %6lld = %lf%%\n", hitnum, hitden, 100.*hitnum/hitden);
#endif
		printf("Bit\tbytes\tratio,\tbytes/bitplane = %d\n", imsize>>3);
		for(int k=0;k<depth;++k)
		{
			int size=load_int_le(out_data+out_idx_sizes+k*sizeof(int));
			printf("%2d\t%5d\t%lf\n", depth-1-k, size, (double)imsize/(size<<3));
		}
		
		printf("Preview:\n");
		int kprint=out_size-out_start<200?(int)(out_size-out_start):200;
		for(int k=0;k<kprint;++k)
			printf("%02X-", out_data[out_start+k]&0xFF);
		printf("\n");
	}
	return true;
}
int				abac4_decode(const unsigned char *in_data, unsigned long long &in_idx, unsigned long long in_size, void *dst, int imsize, int depth, int bytestride, int loud)
{
	auto buffer=(unsigned char*)dst;
	if(!in_data||!imsize||!depth||!bytestride)
		FAIL("abac4_decode(data=%p, imsize=%d, depth=%d, stride=%d)", in_data, imsize, depth, bytestride);
	auto t1=__rdtsc();
	//memset(buffer, 0, imsize*bytestride);

	int headercount=1+(depth<<1);
	if(in_idx+headercount*sizeof(int)>=in_size)
		FAIL("Missing information: idx=%lld, size=%lld", in_idx, in_size);
	int magic=load_int_le(in_data+in_idx);
	if(magic!=magic_ac04)
		FAIL("Invalid magic number 0x%08X, expected 0x%08X", magic, magic_ac04);
	auto sizes=in_data+in_idx+sizeof(int), conf=sizes+depth*sizeof(int), data=conf+depth*sizeof(int);
	in_idx+=headercount*sizeof(int);
	
	int cusize=0;
	for(int kp=depth-1;kp>=0;--kp)//bit-plane loop
	{
		int bit_offset=kp>>3, bit_shift=kp&7;
		int bit_offset2=(kp+1)>>3, bit_shift2=(kp+1)&7;
		int ncodes=load_int_le(sizes+kp*sizeof(int));
	//	int ncodes=load_int_le(sizes+(depth-1-kp)*sizeof(int));
		auto plane=data+cusize;
		
		int prob=0x8000, prob_correct=0x8000;
#if 1
		u64 hitcount=load_int_le(conf+kp*sizeof(int));
	//	u64 hitcount=load_int_le(conf+(depth-1-kp)*sizeof(int));
		if(hitcount<imsize*min_conf)
		{
			for(int kb=0, kb2=0, b=0;kb<imsize;++kb, kb2+=bytestride)
			{
				int byte_idx=kb>>3, bit_idx=kb&7;
				int bit=plane[byte_idx]>>bit_idx&1;
				buffer[kb2+bit_offset]|=bit<<bit_shift;
			//	buffer[kb]|=bit<<kp;
			}
			cusize+=ncodes;
			PROF(DEC_BYPASS_PLANE);
			continue;
		}
#ifdef ABAC2_CONF_MSB_RELATION
		int prevbit0=0;
#endif
		int hitratio_sure=int(0x10000*pow((double)hitcount/imsize, 1/boost_power)), hitratio_notsure=int(0x10000*pow((double)hitcount/imsize, boost_power));
		int hitratio_delta=hitratio_sure-hitratio_notsure;
		//int hitratio_sure=int(0x10000*cbrt((double)hitcount/imsize)), hitratio_notsure=int(0x10000*(double)hitcount*hitcount*hitcount/((double)imsize*imsize*imsize));
		//int hitratio_sure=int(0x10000*sqrt((double)hitcount/imsize)), hitratio_notsure=int(0x10000*(double)hitcount*hitcount/((double)imsize*imsize));
		hitcount=(hitcount<<16)/imsize;
		//hitcount=unsigned(((u64)hitcount<<16)/imsize);
		//hitcount=abac2_normalize16(hitcount, logimsize);
		//hitcount*=invimsize;
#endif

		unsigned start=0;
		u64 range=0xFFFFFFFF;
		unsigned code=load_int_be(plane);
		for(int kc=4, kb=0, kb2=0;kb<imsize;kb2+=bytestride)//bit-pixel loop
		{
			if(range<3)
			{
				code=load_int_be(plane+kc);
				kc+=4;
				start=0, range=0xFFFFFFFF;//because 1=0.9999...
			}
#ifdef ABAC2_CONF_MSB_RELATION
			int prevbit=0;
			if(kp+1<depth)
				prevbit=buffer[kb2+bit_offset2]>>bit_shift2&1;
		//	int prevbit=buffer[kb]>>(kp+1)&1;
#endif
			int p0=prob-0x8000;
			p0=p0*prob_correct>>16;
			p0=p0*prob_correct>>16;
			int sure=-(prevbit==prevbit0);
			p0=p0*(hitratio_notsure+(hitratio_delta&sure))>>16;
			//p0=p0*(prevbit==prevbit0?hitratio_sure:hitratio_notsure)>>16;
			//p0=(long long)p0*hitcount>>16;
			p0+=0x8000;
			//if(prevbit!=prevbit0)
			//	p0=0x8000;
			//	p0=0xFFFF-p0;

			//int p0=0x8000+((long long)(prob-0x8000)*(prevbit==prevbit0?hitratio_sure:hitratio_notsure)>>16);

			//int p0=(long long)(prob-0x8000)*sqrthitcount>>16;
			//if(prevbit==prevbit0)
			//	p0=(long long)p0*hitcount>>16;
			//p0+=0x8000;

			//int confboost=prevbit==prevbit0;
			//confboost-=!confboost;
			//confboost<<=LOG_CONFBOOST;
			//int p0=0x8000+((long long)(prob-0x8000)*(hitcount+confboost)>>16);

		//	int p0=0x8000+(int)((prob-0x8000)*(prevbit==prevbit0?sqrt((double)test_conf[kp]/imsize):(double)test_conf[kp]*test_conf[kp]/((double)imsize*imsize)));
		//	int p0=prevbit==prevbit0?prob:0x8000;
		//	int p0=0x8000+(long long)(prob-0x8000)*test_conf[kp]/imsize;
		//	int p0=0x8000+(long long)(prob-0x8000)*hitcount/(kb+1);
			p0=clamp(1, p0, prob_max);
			unsigned r2=(unsigned)(range*p0>>16);
			r2+=(r2==0)-(r2==range);
			unsigned middle=start+r2;
			int bit=code>middle;
#ifdef DEBUG_ABAC2
			if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
				printf("%6d %6d %d %08X+%08X %08X %08X %08X\n", kp, kb, bit, start, (int)range, r2, middle, code);
#endif
			
			int correct=bit^(p0>=0x8000);
		//	hitcount+=correct;
			prob=!bit<<15|prob>>1;
			prob_correct=correct<<15|prob_correct>>1;
#ifdef ABAC2_CONF_MSB_RELATION
			prevbit0=prevbit;
#endif
			
			if(bit)
			{
				++r2;
				start+=r2, range-=r2;
			}
			//	start=middle+1;
			else
				range=r2-1;
			//	end=middle-1;
			
			buffer[kb2+bit_offset]|=bit<<bit_shift;
		//	buffer[kb]|=bit<<kp;
			++kb;
			
			while((start^(start+(unsigned)range))<0x1000000)//shift-out identical bytes			zpaq 1.10
			{
#ifdef DEBUG_ABAC2
				if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
					printf("range 0x%08X byte-out 0x%02X\n", (int)range, code>>24);
#endif
				code=code<<8|(unsigned char)plane[kc];
				++kc;
				start<<=8;
				range=range<<8|0xFF;
			}
		}
		cusize+=ncodes;
		PROF(DEC_AC);
	}
	in_idx+=cusize;
	auto t2=__rdtsc();

	if(loud)
	{
		printf("AC decode:  %lld cycles, %lf c/byte\n", t2-t1, (double)(t2-t1)/(imsize*depth>>3));
	}
	return true;
}

//API
struct			Buffer
{
	unsigned long long size, cap;
	unsigned char *data;
};
extern "C" API int	encode_bytes(const unsigned char *src, long long srcsize, Buffer *out, int loud)
{
	int success=abac4_encode(src, srcsize, 8, 1, out->data, out->size, out->cap, loud);
	return success;
}
extern "C" API int	decode_bytes(const unsigned char *in_data, unsigned long long in_idx, unsigned long long in_size, unsigned char *dst, int imsize, int loud)
{
	int success=abac4_decode(in_data, in_idx, in_size, dst, imsize, 8, 1, loud);
	return success;
}
extern "C" API void	free_memory(void *p)
{
	free(p);
}
extern "C" API int test(const int *data, long long size, int loud)
{
	Buffer out={};
	int success=abac4_encode(data, size, 8, 4, out.data, out.size, out.cap, loud);
	int csize=out.size;
	free(out.data);
	return csize;
}